// py-ross: Python binding for ROSS (v0)
//
// Exposes a small surface: ross.LP, ross.Msg, ross.BitField, ross.Simulator,
// ross.register_lp_type. Each ROSS LP holds a PyObject* to a user Python
// instance; C trampolines acquire the GIL and dispatch to overridden methods.

#include <nanobind/nanobind.h>
#include <nanobind/trampoline.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/bind_vector.h>

#include <ross.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>
#include <stdexcept>

namespace nb = nanobind;
using namespace nb::literals;

// ---------------------------------------------------------------------------
// Fixed-size message payload (v0: 256 bytes).
// ---------------------------------------------------------------------------
struct Msg {
    uint32_t msg_type;
    uint32_t _pad;
    uint64_t sender_gid;
    uint8_t  scratch[240];
};
static_assert(sizeof(Msg) == 256, "Msg must be 256 bytes");

// ---------------------------------------------------------------------------
// PyLPState lives in each ROSS LP's state vector. Holds a raw PyObject*.
// We do NOT own the GIL when this struct is touched from C; users of
// ->instance must acquire it first.
// ---------------------------------------------------------------------------
struct PyLPState {
    PyObject *instance;   // strong ref; released in c_final
    tw_lpid   gid;
};

// ---------------------------------------------------------------------------
// Forward decl for the LP Python base type and BitField view.
// ---------------------------------------------------------------------------
struct LP;
struct BitField { tw_bf *bf; };

// ---------------------------------------------------------------------------
// Global registry: type-name -> Python class object.
// Populated from Python via register_lp_type("name", Class).
// Read from c_init via the type_map callback (set on Simulator).
// ---------------------------------------------------------------------------
// Owned strong refs (raw PyObject* so we don't destruct nb::object at process exit).
static std::unordered_map<std::string, PyObject *> g_lp_classes;
static PyObject *g_type_map = nullptr;

// Called with GIL held. Drops all strong refs we own.
static void clear_python_refs() {
    for (auto &kv : g_lp_classes) Py_XDECREF(kv.second);
    g_lp_classes.clear();
    Py_XDECREF(g_type_map);
    g_type_map = nullptr;
}

// Optional: track whether we expect optimistic / reverse handlers
static bool g_optimistic_required = false;

// Single tw_lptype shared by every LP. All LPs route through the same
// trampolines; the per-LP behavior comes from the Python instance.
static tw_lptype g_one_lptype;

// The lone "type 0" used in the typemap callback.
static tw_lpid one_typemap(tw_lpid /*gid*/) { return 0; }

// ---------------------------------------------------------------------------
// LP base class. Stores a back-pointer to its tw_lp so it can call
// ROSS APIs (send, rng, now, gid).
// ---------------------------------------------------------------------------
struct LP {
    tw_lp *lp = nullptr;  // set in c_init

    LP() = default;
    virtual ~LP() = default;

    // Hooks (default no-ops). nanobind trampoline forwards these to Python.
    virtual void init()                                       {}
    virtual void pre_run()                                    {}
    virtual void on_event(Msg & /*m*/, double /*now*/)        {}
    virtual void reverse_event(Msg & /*m*/, BitField & /*bf*/) {
        tw_error(TW_LOC, "reverse_event called on an LP that did not override it (optimistic mode requires it)");
    }
    virtual void commit_event(Msg & /*m*/)                    {}
    virtual void final_()                                     {}
};

// nanobind trampoline so Python subclasses can override the virtuals.
struct LP_Tramp : LP {
    NB_TRAMPOLINE(LP, 6);
    void init()                              override { NB_OVERRIDE(init); }
    void pre_run()                           override { NB_OVERRIDE(pre_run); }
    void on_event(Msg &m, double now)        override { NB_OVERRIDE(on_event, m, now); }
    void reverse_event(Msg &m, BitField &bf) override { NB_OVERRIDE(reverse_event, m, bf); }
    void commit_event(Msg &m)                override { NB_OVERRIDE(commit_event, m); }
    void final_()                            override { NB_OVERRIDE_NAME("final", final_); }
};

// Helper: pull the LP* out of a PyLPState-backed sv.
static inline LP * get_lp_from_sv(void *sv) {
    PyLPState *st = static_cast<PyLPState *>(sv);
    if (!st->instance) return nullptr;
    // Borrowed reference, no GIL change here -- caller already holds GIL.
    LP *self = nb::inst_ptr<LP>(nb::handle(st->instance));
    return self;
}

// ---------------------------------------------------------------------------
// Trampolines: ROSS-callable C functions that bounce into Python.
// ---------------------------------------------------------------------------
static void c_init(void *sv, tw_lp *lp) {
    PyLPState *st = static_cast<PyLPState *>(sv);
    st->gid = lp->gid;
    st->instance = nullptr;

    nb::gil_scoped_acquire gil;
    try {
        // Decide which class to instantiate
        if (!g_type_map) {
            tw_error(TW_LOC, "ross.Simulator: type_map not set");
        }
        nb::object name_obj = nb::borrow(g_type_map)(lp->gid);
        std::string name = nb::cast<std::string>(name_obj);
        auto it = g_lp_classes.find(name);
        if (it == g_lp_classes.end()) {
            tw_error(TW_LOC, "ross: unregistered LP type '%s' for gid %llu",
                     name.c_str(), (unsigned long long) lp->gid);
        }
        // Instantiate the Python class (no args).
        nb::object py_inst = nb::borrow(it->second)();
        // Stash a strong ref. We INCREF here; c_final releases.
        Py_INCREF(py_inst.ptr());
        st->instance = py_inst.ptr();
        // Wire back-pointer.
        LP *self = nb::inst_ptr<LP>(py_inst);
        self->lp = lp;
        // Call user's init().
        self->init();
    } catch (nb::python_error &e) {
        e.restore();
        PyErr_Print();
        tw_error(TW_LOC, "ross: Python exception in init() for gid %llu",
                 (unsigned long long) lp->gid);
    }
}

static void c_pre_run(void *sv, tw_lp *lp) {
    (void) lp;
    nb::gil_scoped_acquire gil;
    LP *self = get_lp_from_sv(sv);
    if (!self) return;
    try { self->pre_run(); }
    catch (nb::python_error &e) {
        e.restore(); PyErr_Print();
        tw_error(TW_LOC, "ross: Python exception in pre_run()");
    }
}

static void c_event(void *sv, tw_bf *bf, void *msg, tw_lp *lp) {
    (void) bf; (void) lp;
    nb::gil_scoped_acquire gil;
    LP *self = get_lp_from_sv(sv);
    if (!self) return;
    Msg *m = static_cast<Msg *>(msg);
    double now = TW_STIME_DBL(tw_now(lp));
    try { self->on_event(*m, now); }
    catch (nb::python_error &e) {
        e.restore(); PyErr_Print();
        tw_error(TW_LOC, "ross: Python exception in on_event()");
    }
}

static void c_revent(void *sv, tw_bf *bf, void *msg, tw_lp *lp) {
    (void) lp;
    nb::gil_scoped_acquire gil;
    LP *self = get_lp_from_sv(sv);
    if (!self) return;
    Msg *m = static_cast<Msg *>(msg);
    BitField bfw{bf};
    try { self->reverse_event(*m, bfw); }
    catch (nb::python_error &e) {
        e.restore(); PyErr_Print();
        tw_error(TW_LOC, "ross: Python exception in reverse_event()");
    }
}

static void c_commit(void *sv, tw_bf *bf, void *msg, tw_lp *lp) {
    (void) bf; (void) lp;
    nb::gil_scoped_acquire gil;
    LP *self = get_lp_from_sv(sv);
    if (!self) return;
    Msg *m = static_cast<Msg *>(msg);
    try { self->commit_event(*m); }
    catch (nb::python_error &e) {
        e.restore(); PyErr_Print();
        // Commit is best-effort; don't abort the simulation.
    }
}

static void c_final(void *sv, tw_lp *lp) {
    (void) lp;
    PyLPState *st = static_cast<PyLPState *>(sv);
    if (!st->instance) return;
    nb::gil_scoped_acquire gil;
    try {
        LP *self = nb::inst_ptr<LP>(nb::handle(st->instance));
        self->final_();
    } catch (nb::python_error &e) {
        e.restore(); PyErr_Print();
    }
    Py_DECREF(st->instance);
    st->instance = nullptr;
}

static tw_peid c_map(tw_lpid gid) {
    // LINEAR mapping: gids [r*nlp, (r+1)*nlp) live on rank r.
    return (tw_peid)(gid / g_tw_nlp);
}

// ---------------------------------------------------------------------------
// Simulator: wraps tw_init/tw_define_lps/tw_run/tw_end.
// ---------------------------------------------------------------------------
struct Simulator {
    tw_lpid lps_per_rank;
    double  end_time;
    int     synch;       // 1=seq, 2=cons, 3=opt, 6=rollback-check
    unsigned int nkp;
    std::vector<std::string> extra_args;
    bool ran = false;

    Simulator(tw_lpid lps_per_rank_,
              nb::object type_map,
              const std::string &synch_str,
              double end_time_,
              unsigned int nkp_,
              std::vector<std::string> extra_args_)
        : lps_per_rank(lps_per_rank_),
          end_time(end_time_),
          nkp(nkp_),
          extra_args(std::move(extra_args_))
    {
        if      (synch_str == "sequential")     synch = 1;
        else if (synch_str == "conservative")   synch = 2;
        else if (synch_str == "optimistic")     synch = 3;
        else if (synch_str == "rollback_check") synch = 6;
        else throw std::runtime_error("synch must be sequential/conservative/optimistic/rollback_check");

        Py_XDECREF(g_type_map);
        Py_INCREF(type_map.ptr());
        g_type_map = type_map.ptr();
        g_optimistic_required = (synch == 3 || synch == 6);
    }

    void run() {
        if (ran) throw std::runtime_error("Simulator.run() may only be called once per process (v0)");
        ran = true;

        // Synthesize argv: progname, --synch=, --end=, --nkp=, extras...
        std::vector<std::string> argv_storage;
        argv_storage.emplace_back("py-ross");
        argv_storage.emplace_back("--synch=" + std::to_string(synch));
        argv_storage.emplace_back("--end=" + std::to_string(end_time));
        argv_storage.emplace_back("--nkp=" + std::to_string(nkp));
        for (auto &a : extra_args) argv_storage.push_back(a);

        std::vector<char *> argv_ptrs;
        for (auto &s : argv_storage) argv_ptrs.push_back(const_cast<char *>(s.c_str()));
        int argc = (int) argv_ptrs.size();
        char **argv = argv_ptrs.data();

        // ROSS init
        {
            nb::gil_scoped_release no_gil;
            tw_init(&argc, &argv);
            tw_define_lps(lps_per_rank, sizeof(Msg));
            for (tw_lpid i = 0; i < g_tw_nlp; ++i) {
                tw_lp_settype(i, &g_one_lptype);
            }
            tw_run();
            tw_end();
        }

        // Drop all Python refs we own. Important: this must happen while
        // the interpreter is still alive (i.e. before module shutdown), or
        // the static destructors will touch freed Python state and crash.
        // tw_end() has already invoked our c_final on every LP, so the
        // per-LP PyObject* refs are gone; we just need to release the
        // registry and the type_map callable.
        {
            nb::gil_scoped_acquire gil;
            clear_python_refs();
        }
    }

    ~Simulator() {
        // Belt-and-braces: if the user never called run() (e.g. an exception
        // between ctor and run), still release the type_map ref.
        if (Py_IsInitialized()) {
            nb::gil_scoped_acquire gil;
            clear_python_refs();
        }
    }
};

// ---------------------------------------------------------------------------
// register_lp_type (Python side). Caller has the GIL.
// ---------------------------------------------------------------------------
static void register_lp_type(const std::string &name, nb::object cls) {
    auto it = g_lp_classes.find(name);
    if (it != g_lp_classes.end()) Py_DECREF(it->second);
    Py_INCREF(cls.ptr());
    g_lp_classes[name] = cls.ptr();
}

// ---------------------------------------------------------------------------
// LP method shims (called from Python; we have the GIL).
// ---------------------------------------------------------------------------
static uint64_t lp_get_gid(LP &self) {
    if (!self.lp) throw std::runtime_error("LP not attached yet");
    return (uint64_t) self.lp->gid;
}

static double lp_get_now(LP &self) {
    if (!self.lp) throw std::runtime_error("LP not attached yet");
    return TW_STIME_DBL(tw_now(self.lp));
}

static double lp_rand_uniform(LP &self) {
    if (!self.lp) throw std::runtime_error("LP not attached yet");
    return tw_rand_unif(self.lp->rng);
}

static double lp_rand_exponential(LP &self, double mean) {
    if (!self.lp) throw std::runtime_error("LP not attached yet");
    return tw_rand_exponential(self.lp->rng, mean);
}

static long lp_rand_integer(LP &self, long lo, long hi) {
    if (!self.lp) throw std::runtime_error("LP not attached yet");
    return tw_rand_integer(self.lp->rng, lo, hi);
}

static void lp_rev_rand_uniform(LP &self) {
    if (!self.lp) throw std::runtime_error("LP not attached yet");
    tw_rand_reverse_unif(self.lp->rng);
}

static void lp_rev_rand_exponential(LP &self) {
    if (!self.lp) throw std::runtime_error("LP not attached yet");
    tw_rand_reverse_unif(self.lp->rng);
}

static void lp_rev_rand_integer(LP &self) {
    if (!self.lp) throw std::runtime_error("LP not attached yet");
    tw_rand_reverse_unif(self.lp->rng);
}

// send(dest, ts_offset, msg_type, scratch=b"")
static void lp_send(LP &self, uint64_t dest_gid, double offset,
                    uint32_t msg_type, nb::bytes scratch)
{
    if (!self.lp) throw std::runtime_error("LP not attached yet");
    tw_event *e = tw_event_new((tw_lpid) dest_gid, (tw_stime) offset, self.lp);
    Msg *out = static_cast<Msg *>(tw_event_data(e));
    out->msg_type   = msg_type;
    out->_pad       = 0;
    out->sender_gid = self.lp->gid;
    std::memset(out->scratch, 0, sizeof(out->scratch));
    Py_ssize_t n = (Py_ssize_t) scratch.size();
    if (n > (Py_ssize_t) sizeof(out->scratch)) {
        throw std::runtime_error("ross.LP.send: scratch larger than 240 bytes");
    }
    if (n > 0) std::memcpy(out->scratch, scratch.c_str(), (size_t) n);
    tw_event_send(e);
}

// ---------------------------------------------------------------------------
// BitField: thin view onto tw_bf*. Lets the Python user read/write
// bf.c0 .. bf.c31 in optimistic mode. (Struct declared above.)
// ---------------------------------------------------------------------------
#define BF_PROP(N) \
    .def_prop_rw("c" #N, \
        [](BitField &self){ return (bool) self.bf->c##N; }, \
        [](BitField &self, bool v){ self.bf->c##N = v ? 1u : 0u; })

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------
NB_MODULE(_ross, m) {
    m.doc() = "Python bindings for ROSS (Rensselaer's Optimistic Simulation System)";

    // Install our single shared tw_lptype with all the trampolines.
    g_one_lptype.init       = (init_f)    c_init;
    g_one_lptype.pre_run    = (pre_run_f) c_pre_run;
    g_one_lptype.event      = (event_f)   c_event;
    g_one_lptype.revent     = (revent_f)  c_revent;
    g_one_lptype.commit     = (commit_f)  c_commit;
    g_one_lptype.final      = (final_f)   c_final;
    g_one_lptype.map        = (map_f)     c_map;
    g_one_lptype.state_sz   = sizeof(PyLPState);

    // ---- Msg (a view onto the C payload) --------------------------------
    nb::class_<Msg>(m, "Msg")
        .def_rw("msg_type", &Msg::msg_type)
        .def_rw("sender_gid", &Msg::sender_gid)
        .def_prop_ro("scratch",
            [](Msg &self) {
                return nb::bytes(reinterpret_cast<const char *>(self.scratch),
                                 sizeof(self.scratch));
            },
            "Read-only copy of the 240-byte scratch buffer.");

    // ---- BitField -------------------------------------------------------
    nb::class_<BitField>(m, "BitField")
        BF_PROP(0)  BF_PROP(1)  BF_PROP(2)  BF_PROP(3)
        BF_PROP(4)  BF_PROP(5)  BF_PROP(6)  BF_PROP(7)
        BF_PROP(8)  BF_PROP(9)  BF_PROP(10) BF_PROP(11)
        BF_PROP(12) BF_PROP(13) BF_PROP(14) BF_PROP(15)
        BF_PROP(16) BF_PROP(17) BF_PROP(18) BF_PROP(19)
        BF_PROP(20) BF_PROP(21) BF_PROP(22) BF_PROP(23)
        BF_PROP(24) BF_PROP(25) BF_PROP(26) BF_PROP(27)
        BF_PROP(28) BF_PROP(29) BF_PROP(30) BF_PROP(31)
        ;

    // ---- LP -------------------------------------------------------------
    nb::class_<LP, LP_Tramp>(m, "LP")
        .def(nb::init<>())
        .def("init",          &LP::init)
        .def("pre_run",       &LP::pre_run)
        .def("on_event",      &LP::on_event)
        .def("reverse_event", &LP::reverse_event)
        .def("commit_event",  &LP::commit_event)
        .def("final",         &LP::final_)
        .def_prop_ro("gid",   &lp_get_gid)
        .def_prop_ro("now",   &lp_get_now)
        .def("rand_uniform",        &lp_rand_uniform)
        .def("rand_exponential",    &lp_rand_exponential, "mean"_a)
        .def("rand_integer",        &lp_rand_integer, "lo"_a, "hi"_a)
        .def("rev_rand_uniform",    &lp_rev_rand_uniform)
        .def("rev_rand_exponential",&lp_rev_rand_exponential)
        .def("rev_rand_integer",    &lp_rev_rand_integer)
        .def("send",                &lp_send,
             "dest_gid"_a, "ts_offset"_a, "msg_type"_a,
             "scratch"_a = nb::bytes(""))
        ;

    // ---- Registry -------------------------------------------------------
    m.def("register_lp_type", &register_lp_type, "name"_a, "cls"_a,
          "Register a Python class to instantiate for LPs with this type name.");

    // ---- Simulator ------------------------------------------------------
    nb::class_<Simulator>(m, "Simulator")
        .def(nb::init<tw_lpid, nb::object, const std::string &, double,
                      unsigned int, std::vector<std::string>>(),
             "lps_per_rank"_a,
             "type_map"_a,
             "synch"_a = "conservative",
             "end_time"_a = 100.0,
             "nkp"_a = 16,
             "extra_args"_a = std::vector<std::string>{})
        .def("run", &Simulator::run);
}
