// py-ross: Python binding for ROSS.
//
// Exposes ross.LP, ross.Msg, ross.BitField, ross.Simulator,
// ross.register_lp_type, ross.PAYLOAD_BYTES. Each ROSS LP holds a PyObject*
// to a user Python instance; C trampolines acquire the GIL and dispatch to
// overridden methods.
//
// Message payload: a fixed POD with an 8-byte runtime header
// {sender_gid, payload_len} followed by a 4 KiB pickle buffer. The Python
// `Msg` exposes only `payload`, a lazily-unpickled object (or None). The
// sender's gid is delivered to handlers as a positional `sender: int` arg
// rather than as a field on `Msg`.

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
// Message payload layout.
//
// The on-wire/in-pool object is a `MsgHeader` immediately followed by a byte
// buffer sized at Simulator construction time. ROSS sees one fixed size
// (header + g_max_msg_size) via tw_define_lps. We never declare the buffer
// as part of the C++ struct — we address it as `reinterpret_cast<uint8_t*>(
// hdr + 1)`, valid for [0, g_max_msg_size).
//
// The Python-visible `Msg` is a thin view {hdr*, cap} built by the trampolines
// on the stack and passed to user handlers by reference.
// ---------------------------------------------------------------------------
static constexpr std::size_t DEFAULT_MAX_MSG_SIZE = 256;

struct MsgHeader {
    uint64_t sender_gid;             // sender LP gid (delivered to handler as `sender`)
    uint32_t payload_len;            // bytes of pickle stream after the header, or 0
    uint32_t _pad;                   // keep buffer 8-byte aligned
};
static_assert(sizeof(MsgHeader) == 16, "MsgHeader must be 16 bytes");

struct Msg {
    MsgHeader  *hdr;
    std::size_t cap;                 // == g_max_msg_size at construction time
    uint8_t * buf()       { return reinterpret_cast<uint8_t *>(hdr + 1); }
    const uint8_t * buf() const { return reinterpret_cast<const uint8_t *>(hdr + 1); }
};

// Active payload-buffer ceiling. Set at Simulator::run() before tw_define_lps
// and read by trampolines / lp_send thereafter. v0 forbids re-entering
// Simulator() in the same process, so there is no concurrency hazard.
static std::size_t g_max_msg_size = 0;

// ---------------------------------------------------------------------------
// PyLPState lives in each ROSS LP's state vector. Holds a raw PyObject*.
// ---------------------------------------------------------------------------
struct PyLPState {
    PyObject *instance;   // strong ref; released in c_final
    tw_lpid   gid;
};

// ---------------------------------------------------------------------------
// Forward decls.
// ---------------------------------------------------------------------------
struct LP;
struct BitField { tw_bf *bf; };

// ---------------------------------------------------------------------------
// Globals owned via raw PyObject* (so we don't destruct nb::object at process
// exit, after the interpreter has finalized).
// ---------------------------------------------------------------------------
static std::unordered_map<std::string, PyObject *> g_lp_classes;
static PyObject *g_type_map     = nullptr;
static PyObject *g_pickle_dumps = nullptr;  // pickle.dumps
static PyObject *g_pickle_loads = nullptr;  // pickle.loads
static PyObject *g_pickle_protocol = nullptr;  // int: HIGHEST_PROTOCOL

// Called with GIL held.
static void clear_python_refs() {
    for (auto &kv : g_lp_classes) Py_XDECREF(kv.second);
    g_lp_classes.clear();
    Py_XDECREF(g_type_map);       g_type_map = nullptr;
    Py_XDECREF(g_pickle_dumps);   g_pickle_dumps = nullptr;
    Py_XDECREF(g_pickle_loads);   g_pickle_loads = nullptr;
    Py_XDECREF(g_pickle_protocol); g_pickle_protocol = nullptr;
}

// Called with GIL held. Idempotent.
static void ensure_pickle_loaded() {
    if (g_pickle_dumps && g_pickle_loads && g_pickle_protocol) return;
    PyObject *mod = PyImport_ImportModule("pickle");
    if (!mod) throw nb::python_error();
    g_pickle_dumps = PyObject_GetAttrString(mod, "dumps");
    g_pickle_loads = PyObject_GetAttrString(mod, "loads");
    g_pickle_protocol = PyObject_GetAttrString(mod, "HIGHEST_PROTOCOL");
    Py_DECREF(mod);
    if (!g_pickle_dumps || !g_pickle_loads || !g_pickle_protocol) {
        throw nb::python_error();
    }
}

static bool g_optimistic_required = false;

// Single tw_lptype shared by every LP. All LPs route through these trampolines;
// per-LP behavior comes from the Python instance.
static tw_lptype g_one_lptype;

// ---------------------------------------------------------------------------
// LP base class.
// ---------------------------------------------------------------------------
struct LP {
    tw_lp *lp = nullptr;

    LP() = default;
    virtual ~LP() = default;

    virtual void init()                                                        {}
    virtual void pre_run()                                                     {}
    virtual void on_event(uint64_t /*sender*/, Msg & /*m*/, double /*now*/)    {}
    virtual void reverse_event(uint64_t /*sender*/, Msg & /*m*/, BitField & /*bf*/) {
        tw_error(TW_LOC, "reverse_event called on an LP that did not override it (optimistic mode requires it)");
    }
    virtual void commit_event(uint64_t /*sender*/, Msg & /*m*/)                {}
    virtual void final_()                                                      {}
};

struct LP_Tramp : LP {
    NB_TRAMPOLINE(LP, 6);
    void init()                                                  override { NB_OVERRIDE(init); }
    void pre_run()                                               override { NB_OVERRIDE(pre_run); }
    void on_event(uint64_t sender, Msg &m, double now)           override { NB_OVERRIDE(on_event, sender, m, now); }
    void reverse_event(uint64_t sender, Msg &m, BitField &bf)    override { NB_OVERRIDE(reverse_event, sender, m, bf); }
    void commit_event(uint64_t sender, Msg &m)                   override { NB_OVERRIDE(commit_event, sender, m); }
    void final_()                                                override { NB_OVERRIDE_NAME("final", final_); }
};

static inline LP * get_lp_from_sv(void *sv) {
    PyLPState *st = static_cast<PyLPState *>(sv);
    if (!st->instance) return nullptr;
    return nb::inst_ptr<LP>(nb::handle(st->instance));
}

// ---------------------------------------------------------------------------
// Trampolines.
// ---------------------------------------------------------------------------
static void c_init(void *sv, tw_lp *lp) {
    PyLPState *st = static_cast<PyLPState *>(sv);
    st->gid = lp->gid;
    st->instance = nullptr;

    nb::gil_scoped_acquire gil;
    try {
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
        nb::object py_inst = nb::borrow(it->second)();
        Py_INCREF(py_inst.ptr());
        st->instance = py_inst.ptr();
        LP *self = nb::inst_ptr<LP>(py_inst);
        self->lp = lp;
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
    (void) bf;
    nb::gil_scoped_acquire gil;
    LP *self = get_lp_from_sv(sv);
    if (!self) return;
    MsgHeader *h = static_cast<MsgHeader *>(msg);
    Msg m{h, g_max_msg_size};
    double now = TW_STIME_DBL(tw_now(lp));
    try { self->on_event(h->sender_gid, m, now); }
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
    MsgHeader *h = static_cast<MsgHeader *>(msg);
    Msg m{h, g_max_msg_size};
    BitField bfw{bf};
    try { self->reverse_event(h->sender_gid, m, bfw); }
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
    MsgHeader *h = static_cast<MsgHeader *>(msg);
    Msg m{h, g_max_msg_size};
    try { self->commit_event(h->sender_gid, m); }
    catch (nb::python_error &e) {
        e.restore(); PyErr_Print();
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
    return (tw_peid)(gid / g_tw_nlp);
}

// ---------------------------------------------------------------------------
// Simulator.
// ---------------------------------------------------------------------------
struct Simulator {
    tw_lpid lps_per_rank;
    double  end_time;
    int     synch;
    unsigned int nkp;
    std::size_t max_msg_size;
    std::vector<std::string> extra_args;
    bool ran = false;

    Simulator(tw_lpid lps_per_rank_,
              nb::object type_map,
              const std::string &synch_str,
              double end_time_,
              unsigned int nkp_,
              std::size_t max_msg_size_,
              std::vector<std::string> extra_args_)
        : lps_per_rank(lps_per_rank_),
          end_time(end_time_),
          nkp(nkp_),
          max_msg_size(max_msg_size_),
          extra_args(std::move(extra_args_))
    {
        if      (synch_str == "sequential")     synch = 1;
        else if (synch_str == "conservative")   synch = 2;
        else if (synch_str == "optimistic")     synch = 3;
        else if (synch_str == "rollback_check") synch = 6;
        else throw std::runtime_error("synch must be sequential/conservative/optimistic/rollback_check");

        if (max_msg_size == 0) {
            throw std::runtime_error("ross.Simulator: max_msg_size must be > 0");
        }

        Py_XDECREF(g_type_map);
        Py_INCREF(type_map.ptr());
        g_type_map = type_map.ptr();
        g_optimistic_required = (synch == 3 || synch == 6);
    }

    void run() {
        if (ran) throw std::runtime_error("Simulator.run() may only be called once per process");
        ran = true;

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

        // Make sure pickle is importable before we release the GIL; we'll
        // need it inside c_event for any send/receive that carries a payload.
        ensure_pickle_loaded();

        // Publish the buffer-ceiling globally so trampolines / lp_send can
        // see it. Must happen before tw_define_lps allocates the event pool.
        g_max_msg_size = max_msg_size;

        {
            nb::gil_scoped_release no_gil;
            tw_init(&argc, &argv);
            tw_define_lps(lps_per_rank, sizeof(MsgHeader) + max_msg_size);
            for (tw_lpid i = 0; i < g_tw_nlp; ++i) {
                tw_lp_settype(i, &g_one_lptype);
            }
            tw_run();
            tw_end();
        }

        {
            nb::gil_scoped_acquire gil;
            clear_python_refs();
        }
    }

    ~Simulator() {
        if (Py_IsInitialized()) {
            nb::gil_scoped_acquire gil;
            clear_python_refs();
        }
    }
};

// ---------------------------------------------------------------------------
// register_lp_type.
// ---------------------------------------------------------------------------
static void register_lp_type(const std::string &name, nb::object cls) {
    auto it = g_lp_classes.find(name);
    if (it != g_lp_classes.end()) Py_DECREF(it->second);
    Py_INCREF(cls.ptr());
    g_lp_classes[name] = cls.ptr();
}

// ---------------------------------------------------------------------------
// LP method shims.
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

// send(dest_gid, ts_offset, payload=None)
// If payload is None, sends an empty event. Otherwise pickles `payload` and
// copies it into the outgoing Msg. Raises OverflowError if the pickled
// representation exceeds the active Simulator's max_msg_size.
static void lp_send(LP &self, uint64_t dest_gid, double offset,
                    nb::object payload)
{
    if (!self.lp) throw std::runtime_error("LP not attached yet");

    tw_event *e = tw_event_new((tw_lpid) dest_gid, (tw_stime) offset, self.lp);
    MsgHeader *out = static_cast<MsgHeader *>(tw_event_data(e));
    out->sender_gid = self.lp->gid;
    out->_pad       = 0;

    if (payload.is_none()) {
        out->payload_len = 0;
        // Leave the buffer uninitialised — receivers must not read it when
        // payload_len == 0. (Avoids a per-empty-event memset.)
        tw_event_send(e);
        return;
    }

    ensure_pickle_loaded();

    // pickle.dumps(payload, HIGHEST_PROTOCOL)
    PyObject *args = PyTuple_Pack(2, payload.ptr(), g_pickle_protocol);
    if (!args) throw nb::python_error();
    PyObject *res = PyObject_Call(g_pickle_dumps, args, nullptr);
    Py_DECREF(args);
    if (!res) throw nb::python_error();

    char *buf = nullptr;
    Py_ssize_t n = 0;
    if (PyBytes_AsStringAndSize(res, &buf, &n) != 0) {
        Py_DECREF(res);
        throw nb::python_error();
    }

    if ((std::size_t) n > g_max_msg_size) {
        Py_DECREF(res);
        std::string msg = "ross.LP.send: pickled payload of "
                        + std::to_string((long long) n)
                        + " bytes exceeds Simulator.max_msg_size ("
                        + std::to_string(g_max_msg_size) + ") limit";
        PyErr_SetString(PyExc_OverflowError, msg.c_str());
        throw nb::python_error();
    }

    std::memcpy(reinterpret_cast<uint8_t *>(out + 1), buf, (std::size_t) n);
    out->payload_len = (uint32_t) n;
    Py_DECREF(res);

    tw_event_send(e);
}

// Msg.payload -> object | None
// Lazily unpickles the buffer. Returns None for empty events.
static nb::object msg_get_payload(Msg &self) {
    if (self.hdr->payload_len == 0) return nb::none();
    ensure_pickle_loaded();
    PyObject *buf = PyBytes_FromStringAndSize(
        reinterpret_cast<const char *>(self.buf()),
        (Py_ssize_t) self.hdr->payload_len);
    if (!buf) throw nb::python_error();
    PyObject *args = PyTuple_Pack(1, buf);
    Py_DECREF(buf);
    if (!args) throw nb::python_error();
    PyObject *res = PyObject_Call(g_pickle_loads, args, nullptr);
    Py_DECREF(args);
    if (!res) throw nb::python_error();
    return nb::steal(res);
}

// ---------------------------------------------------------------------------
// BitField property macro.
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

    g_one_lptype.init       = (init_f)    c_init;
    g_one_lptype.pre_run    = (pre_run_f) c_pre_run;
    g_one_lptype.event      = (event_f)   c_event;
    g_one_lptype.revent     = (revent_f)  c_revent;
    g_one_lptype.commit     = (commit_f)  c_commit;
    g_one_lptype.final      = (final_f)   c_final;
    g_one_lptype.map        = (map_f)     c_map;
    g_one_lptype.state_sz   = sizeof(PyLPState);

    m.attr("DEFAULT_MAX_MSG_SIZE") = (int) DEFAULT_MAX_MSG_SIZE;

    // ---- Msg ------------------------------------------------------------
    nb::class_<Msg>(m, "Msg",
        "Event payload view. Use `msg.payload` to get the unpickled object\n"
        "(or None for an empty event). The sender's LP gid is delivered as a\n"
        "separate `sender` argument to event handlers, not as a Msg field.")
        .def_prop_ro("payload", &msg_get_payload,
            "The unpickled Python object the sender attached, or None if the\n"
            "event was sent with payload=None (an empty event).");

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
             "dest_gid"_a, "ts_offset"_a, "payload"_a = nb::none(),
             "Schedule an event at dest_gid `ts_offset` virtual-time units ahead.\n"
             "If `payload` is given, it is pickled and must round-trip via pickle on\n"
             "the receiver. The class of the payload object must be importable on\n"
             "every rank for cross-rank sends. Raises OverflowError if the pickled\n"
             "payload exceeds the active Simulator's max_msg_size.")
        ;

    // ---- Registry -------------------------------------------------------
    m.def("register_lp_type", &register_lp_type, "name"_a, "cls"_a,
          "Register a Python class to instantiate for LPs with this type name.");

    // ---- Simulator ------------------------------------------------------
    nb::class_<Simulator>(m, "Simulator")
        .def(nb::init<tw_lpid, nb::object, const std::string &, double,
                      unsigned int, std::size_t, std::vector<std::string>>(),
             "lps_per_rank"_a,
             "type_map"_a,
             "synch"_a = "conservative",
             "end_time"_a = 100.0,
             "nkp"_a = 16,
             "max_msg_size"_a = DEFAULT_MAX_MSG_SIZE,
             "extra_args"_a = std::vector<std::string>{})
        .def_ro("max_msg_size", &Simulator::max_msg_size,
                "The per-event payload-buffer ceiling (bytes) chosen at ctor time.")
        .def("run", &Simulator::run);
}
