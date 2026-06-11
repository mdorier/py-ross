# py-ross examples

Five runnable examples, ordered from "smallest binding smoke test" to "realistic FL protocol". Each is a single file; each prints something useful at the end.

All examples accept `--synch={sequential,conservative,optimistic,rollback_check}` and run under MPI via `mpirun -np N python examples/<file>.py ...`. Set `PYTHONPATH=python` (relative to the project root) so the freshly-built extension is importable.

## `phold.py` — the PDES standard benchmark

Port of the canonical PHOLD model from `ROSS/models/phold/phold.main.c`. Each LP receives empty events, draws a random destination (self with probability `1-remote`, a random remote LP otherwise), and forwards one new event after an exponentially-distributed delay. No payload — the arrival itself is the signal.

What it exercises:

- Empty events (`self.send(dest, dt)` with no payload).
- `BitField`-driven branch tracking in `reverse_event` (`bf.c0` remembers whether the remote-LP branch was taken, so the reverse handler knows whether to call `rev_rand_integer()`).
- Cross-rank traffic in all sync modes.

```bash
PYTHONPATH=python python examples/phold.py --synch=sequential --lps-per-rank=8 --end=100
PYTHONPATH=python mpirun -np 4 python examples/phold.py --synch=optimistic --lps-per-rank=8 --end=100 --extra=--extramem=10000
```

## `airport.py` — dataclass-discriminated events

Towers exchange `Arrive` and `Land` events. Each tower queues arriving planes, services them with a queue-length-dependent delay, then dispatches departing planes to a random other tower. Two payload types as module-scope `@dataclass`es, discriminated by `isinstance(msg.payload, Arrive)`.

What it exercises:

- The canonical "dispatch on payload type" idiom that replaces the old `msg_type` integer tag.
- A docstring snippet showing the "reply to `sender`" pattern (`self.send(sender, dt, payload=Ack())`) that the `sender` argument enables.

```bash
PYTHONPATH=python python examples/airport.py --synch=sequential --towers-per-rank=4 --end=50
PYTHONPATH=python mpirun -np 4 python examples/airport.py --synch=conservative --towers-per-rank=4 --end=50
```

## `pingpong.py` — `sender` round-trip check

Two LPs bounce a `Ping(hops, trace)` payload back and forth. Each LP records what `sender` values it saw and asserts at `final()` that it only ever heard from the *other* LP — a regression test for the C++ trampoline wiring of the `sender` argument.

What it exercises:

- The `sender` parameter on event handlers (gid of whichever LP sent the event).
- A small pickled payload that mutates across the round trip (the `trace` list grows by one gid per hop).

```bash
PYTHONPATH=python python examples/pingpong.py --target=10 --synch=sequential
PYTHONPATH=python mpirun -np 2 python examples/pingpong.py --target=10 --synch=conservative
```

## `big_pingpong.py` — large MPI payloads

Two LPs ping-pong a `--payload-bytes`-sized `bytes` blob (default 32 KiB) several times. Validates that the on-wire MPI path correctly transports large `Msg` buffers and that `max_msg_size` can be cranked up at runtime.

What it exercises:

- `Simulator(max_msg_size=BYTES + 256)` — runtime-picked event buffer, much larger than the default 256 B.
- Content integrity: the receiver checks both the size and a couple of bytes of the round-tripped payload.

```bash
PYTHONPATH=python mpirun -np 2 python examples/big_pingpong.py --payload-bytes=32768 --target=4 --synch=conservative
```

## `fedasync.py` — cost-model federated learning

A no-training port of [AFL-Lib](https://github.com/anonyresearch/AFL-Lib)'s `alg/fedasync.py`. One Server LP coordinates `N` Client LPs in an async-FL protocol: server seeds `MAX_CONCURRENCY` initial Pulls, each client trains for `exp(BASE) × device_speed` virtual seconds (no actual ML), each `Done` increments the server's `global_version` and triggers the next round-robin Pull. The server logs `(now, client, version, staleness)` per aggregation.

Compute and bandwidth tiers are copied from AFL-Lib's `utils/sys_utils.py` (TX2/Nano/RPi compute speeds; WiFi/4G/5G bandwidths). The "model" itself never goes on the wire — only a version counter — but a `MODEL_BYTES` knob (default 1 MiB) drives the comm-time latency, mirroring what real FL would pay.

What it exercises:

- A realistic two-LP-type simulation (`type_map = lambda gid: "server" if gid == 0 else "client"`).
- The full async-FL protocol structure end-to-end across MPI ranks.
- Determinism across sync modes: sequential, 2-rank conservative, and 2-rank optimistic all produce identical final state.
- `BitField` with two flags (`bf.c0` = scheduled another Pull, `bf.c1` = crossed the round cap) to handle the "Pulls in flight when the cap is reached" corner case.

```bash
PYTHONPATH=python python examples/fedasync.py --num-clients=7 --rounds=20 --synch=sequential
PYTHONPATH=python mpirun -np 2 python examples/fedasync.py --num-clients=7 --rounds=20 --synch=conservative
PYTHONPATH=python mpirun -np 4 python examples/fedasync.py --num-clients=63 --rounds=200 --synch=conservative
```

The final line names the rounds completed, total virtual wall-clock, mean staleness, and per-client `Done` counts:

```
[server] rounds=20 virtual_t=25.13s mean_staleness=2.20 per_client_dones={1: 2, 2: 4, 3: 4, 4: 4, 5: 2, 6: 2, 7: 2}
```

## A note on MPI rank counts

All examples use the default LINEAR LP mapping: rank `r` owns gids `[r*lps_per_rank, (r+1)*lps_per_rank)`. So the total LP count must divide evenly by the number of MPI ranks. The examples enforce this at startup (`fedasync.py` checks; the others rely on you picking `--lps-per-rank` and `-np` consistently).
