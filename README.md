# py-ross

Python bindings for [ROSS](https://github.com/ROSS-org/ROSS) (Rensselaer's Optimistic Simulation System), a parallel discrete-event simulator with MPI scale-out and Time Warp optimistic rollback. py-ross lets you write LPs (logical processes) as plain Python classes and run them under sequential, conservative, or optimistic synchronization on a single core or across MPI ranks.

```python
import ross
from dataclasses import dataclass

@dataclass
class Hello:
    text: str

@ross.lp("greeter")
class Greeter(ross.LP):
    def init(self):
        if self.gid == 0:
            self.send(1, 1.0, payload=Hello(text="hi from 0"))

    def on_event(self, sender, msg, now, bf):
        print(f"LP {self.gid} got {msg.payload.text!r} from {sender} at t={now}")

sim = ross.Simulator(
    lps_per_rank=2,
    type_map=lambda gid: "greeter",
    synch="sequential",
    end_time=10.0,
    max_msg_size=256,
)
sim.run()
```

## Why

ROSS is fast and battle-tested but requires LPs to be written in C, with `tw_lptype` vtables of function pointers and hand-rolled reverse handlers. That excludes most ML researchers, who want to write event-driven models (federated learning, distributed training, scheduling) in Python without giving up the parallel runtime. py-ross wraps the ROSS C API with [nanobind](https://github.com/wjakko/nanobind) and exposes a small, Pythonic surface: a `ross.LP` base class with lifecycle hooks, pickle-based event payloads, and a `Simulator` driver.

## Status

Working: sequential / conservative / optimistic / rollback-check sync modes; single-rank and multi-rank MPI; pickled payloads with runtime-configurable buffer size; reverse handlers with per-event `BitField` for branch tracking; clean teardown (no nanobind leaks on optimistic-mode shutdown).

Not yet implemented yet:
- **Variable-size payloads / out-of-band transport.** `Simulator(max_msg_size=N)` is a hard cap; everything ships at that size. This is because ROSS expects every message to be the same size.
- **Multiple `Simulator()` instances per process.** ROSS's globals + `MPI_Init`/`MPI_Finalize` lifecycle make this unfeasible without upstream patches. Subprocess-per-run is the supported pattern.
- **Custom LP mappings.** v0 uses LINEAR only (`rank r` owns gids `[r*lps_per_rank, (r+1)*lps_per_rank)`).

## Installation

py-ross depends on:

- ROSS, built and installed (provides `ROSSConfig.cmake`).
- An MPI implementation (mpich or openmpi); the one ROSS was built against.
- Python ≥ 3.9 with development headers.
- nanobind, scikit-build-core, cmake ≥ 3.18, a C++17 compiler.

The recommended way to provision these is via [Spack](https://spack.io/). A `spack.yaml` is included for reference; activating that environment gets you a working toolchain plus a built ROSS:

```bash
spack env create py-ross-env spack.yaml
spack env activate py-ross-env
spack install
```

Build py-ross out of tree:

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
cp build/_ross*.so python/ross/
```

(There's no `pip install .` recipe yet — that requires wiring the spack toolchain into scikit-build-core. For now we run with `PYTHONPATH=python`.)

## Concepts

### LPs as decorated classes

```python
@ross.lp("my_lp_type")
class MyLP(ross.LP):
    def init(self): ...
    def pre_run(self): ...
    def on_event(self, sender, msg, now, bf): ...
    def reverse_event(self, sender, msg, bf): ...   # required for optimistic
    def commit_event(self, sender, msg, bf): ...
    def final(self): ...
```

The decorator registers the class under a name. At simulator construction time you pass a `type_map: Callable[[int gid], str]` that picks the LP type per gid. The base class provides `self.gid`, `self.now`, `self.send(dest, ts_offset, payload=None)`, and reversible RNG (`rand_uniform`, `rand_exponential`, `rand_integer`, plus `rev_rand_*` for reverse handlers).

### Payloads

`self.send(dest, ts_offset, payload=obj)` pickles `obj` into the event. The receiver gets it back via `msg.payload`. Empty events (no payload) skip pickle entirely.

The payload class must be importable on every MPI rank (the receiving rank unpickles in its own interpreter). Define payload classes at module scope, not inside functions.

```python
@dataclass
class TaskDone:
    worker_id: int
    result: float

self.send(server_gid, 1.5, payload=TaskDone(worker_id=self.gid, result=0.42))
```

### `BitField`

Each event carries a 32-bit `tw_bf` that you can use to remember branch decisions made in `on_event`, so `reverse_event` knows what to undo:

```python
def on_event(self, sender, msg, now, bf):
    if self.rand_uniform() < 0.5:
        bf.c0 = True            # took the "send a reply" branch
        self.send(sender, 1.0, payload=Ack())
    else:
        bf.c0 = False

def reverse_event(self, sender, msg, bf):
    self.rev_rand_uniform()
    # The send itself is undone by ROSS; we just track our own state.
```

### Sync modes

- `"sequential"` — single core, one event at a time. Reverse handlers never run. Fastest for debugging.
- `"conservative"` — parallel, no rollback. Requires events to carry strictly positive lookahead (`ts_offset >= g_tw_lookahead`, default 0.005).
- `"optimistic"` — Time Warp. Reverse handlers must undo every state mutation and RNG draw made in `on_event`.
- `"rollback_check"` — runs every event forward then immediately reverse-executes, useful for catching reverse-handler bugs. (Requires a ROSS build with `--with-rollback-check`; the spack default doesn't include it.)

## See also

- `examples/README.md` — what each example demonstrates.
- `python/ross/_ross.pyi` — full type stubs for the C extension (read by mypy and IDEs).
- ROSS upstream: https://github.com/ROSS-org/ROSS
