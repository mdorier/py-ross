"""Cost-model FedAsync on py-ross.

A no-training port of AFL-Lib's `alg/fedasync.py`. The protocol is real;
the "training" is a sampled compute delay and the "model" is just a version
counter. Demonstrates that py-ross can run a recognizable async-FL protocol
end-to-end across MPI ranks without any ML dependency.

Topology:
  gid 0          -> Server (one LP, owns global_version + a pool of idle clients)
  gid 1 .. N     -> Client LPs

Protocol (mirrors AFL-Lib):
  Server.init()        : seeds K = MAX_CONCURRENCY initial Pull(v=0) to clients
  Client.on_event Pull : schedules a Done(v) to the server after
                         comm_down + compute(exp(BASE)*device_speed) + comm_up
  Server.on_event Done : staleness = global_version - v
                         global_version += 1; log a row
                         if rounds remaining, pick an idle client and send Pull

The "model" itself never goes on the wire. `Pull(version)` and `Done(version)`
carry only the version number (each pickles to ~30 B, so max_msg_size=64 is
plenty). `MODEL_BYTES` exists only to derive the comm-time latency.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from typing import List, Tuple

import ross

# ---------------------------------------------------------------------------
# Cost model: tiers copied from AFL-Lib's utils/sys_utils.py.
# device_speed multiplies the exponential base compute time.
# ---------------------------------------------------------------------------
DEVICE_TIERS: List[Tuple[str, float]] = [
    ("TX2",  1.0),
    ("Nano", 1.8),
    ("RPi",  11.6),
]
BANDWIDTH_MBPS_TIERS: List[Tuple[str, float]] = [
    ("WiFi", 300.0),
    ("4G",    60.0),
    ("5G",   500.0),
]

# Runtime knobs (set in main() before sim.run()).
NUM_CLIENTS: int = 0
TOTAL_ROUNDS: int = 0
MAX_CONCURRENCY: int = 0
COMPUTE_MEAN_BASE: float = 1.0
MODEL_BYTES: int = 1_048_576   # 1 MiB; only used to derive comm latency

# A tiny constant added to every server -> client send so that events always
# carry strictly positive lookahead. ROSS conservative mode rejects events
# with ts_offset < g_tw_lookahead (default 0.005). The client's own
# Done-back-to-server already carries a much larger compute+comm delay, so
# only the server's Pulls need this nudge.
DISPATCH_DELAY: float = 0.01


# ---------------------------------------------------------------------------
# Message payload classes (module-scope so they pickle/unpickle across ranks).
# ---------------------------------------------------------------------------
@dataclass
class Pull:
    """Server -> Client: 'please train against this global version'."""
    version: int


@dataclass
class Done:
    """Client -> Server: 'I finished training against this version'."""
    version: int


# ---------------------------------------------------------------------------
# Server LP. Single instance at gid 0.
# ---------------------------------------------------------------------------
@ross.lp("server")
class Server(ross.LP):
    global_version: int
    in_flight: int                              # how many Pulls currently outstanding
    history: List[Tuple[float, int, int, int]]  # (now, client_gid, version, staleness)

    def init(self) -> None:
        self.global_version = 0
        self.in_flight = 0
        self.history = []
        # Seed the initial round: send a Pull(v=0) to the first K clients.
        # We charge a small constant "downlink dispatch" delay so events
        # always carry strictly positive lookahead — conservative-mode
        # safety. Real downlink/compute/uplink cost is paid by the client.
        for k in range(MAX_CONCURRENCY):
            client_gid = 1 + (k % NUM_CLIENTS)
            self.send(client_gid, DISPATCH_DELAY, payload=Pull(version=0))
            self.in_flight += 1

    def on_event(self, sender: int, msg: ross.Msg, now: float,
                 bf: ross.BitField) -> None:
        p = msg.payload
        assert isinstance(p, Done), f"server expected Done, got {type(p).__name__}"
        # bf.c1: "did we log this Done?"   (false once we've hit the cap)
        # bf.c0: "did we schedule another Pull?"
        if self.global_version >= TOTAL_ROUNDS:
            bf.c0 = False
            bf.c1 = False
            self.in_flight -= 1
            return
        bf.c1 = True
        staleness = self.global_version - p.version
        self.history.append((now, sender, p.version, staleness))
        self.global_version += 1
        self.in_flight -= 1

        # Schedule the next Pull unless this very Done crossed the cap.
        if self.global_version < TOTAL_ROUNDS:
            bf.c0 = True
            next_client = 1 + (self.global_version % NUM_CLIENTS)
            self.send(next_client, DISPATCH_DELAY,
                      payload=Pull(version=self.global_version))
            self.in_flight += 1
        else:
            bf.c0 = False

    def reverse_event(self, sender: int, msg: ross.Msg,
                      bf: ross.BitField) -> None:
        # Undo in reverse order of the forward effects.
        if not bf.c1:
            # We hit the cap branch on the way forward — only in_flight changed.
            self.in_flight += 1
            return
        if bf.c0:
            self.in_flight -= 1
        self.in_flight += 1
        self.global_version -= 1
        self.history.pop()

    def final(self) -> None:
        if not self.history:
            return
        rounds = len(self.history)
        end_time = self.history[-1][0]
        mean_stale = sum(s for _, _, _, s in self.history) / rounds
        per_client: dict[int, int] = {}
        for _, c, _, _ in self.history:
            per_client[c] = per_client.get(c, 0) + 1
        print(f"[server] rounds={rounds} virtual_t={end_time:.2f}s "
              f"mean_staleness={mean_stale:.2f} "
              f"per_client_dones={dict(sorted(per_client.items()))}")


# ---------------------------------------------------------------------------
# Client LP. One instance at gid 1..N.
# ---------------------------------------------------------------------------
@ross.lp("client")
class Client(ross.LP):
    device_speed: float
    bandwidth_mbps: float

    def init(self) -> None:
        # Assign tiers once at init using ROSS RNG (per-LP, reversible).
        d_idx = int(self.rand_integer(0, len(DEVICE_TIERS) - 1))
        b_idx = int(self.rand_integer(0, len(BANDWIDTH_MBPS_TIERS) - 1))
        self.device_speed = DEVICE_TIERS[d_idx][1]
        self.bandwidth_mbps = BANDWIDTH_MBPS_TIERS[b_idx][1]

    def _comm_time(self) -> float:
        # bytes -> bits / (Mbps -> bits/s)
        return (MODEL_BYTES * 8) / (self.bandwidth_mbps * 1e6)

    def on_event(self, sender: int, msg: ross.Msg, now: float,
                 bf: ross.BitField) -> None:
        p = msg.payload
        assert isinstance(p, Pull), f"client expected Pull, got {type(p).__name__}"
        compute = self.rand_exponential(COMPUTE_MEAN_BASE) * self.device_speed
        round_trip = self._comm_time() + compute + self._comm_time()
        # Reply to whichever server sent the Pull (supports future multi-server).
        self.send(sender, round_trip, payload=Done(version=p.version))

    def reverse_event(self, sender: int, msg: ross.Msg,
                      bf: ross.BitField) -> None:
        self.rev_rand_exponential()


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--num-clients", type=int, default=7,
                   help="Total client LPs; total LPs = num_clients + 1 (server)")
    p.add_argument("--rounds", type=int, default=20,
                   help="Number of aggregation rounds before the server stops")
    p.add_argument("--concurrency", type=int, default=4,
                   help="MAX_CONCURRENCY: initial Pulls in flight")
    p.add_argument("--compute-mean", type=float, default=1.0,
                   help="Mean of the exponential compute-time distribution (seconds)")
    p.add_argument("--model-bytes", type=int, default=1_048_576,
                   help="Bytes used to derive comm-time latency (not actually sent)")
    p.add_argument("--synch", default="sequential",
                   choices=["sequential", "conservative",
                            "optimistic", "rollback_check"])
    p.add_argument("--end", type=float, default=1e6,
                   help="Simulation end virtual time (large; server stops earlier)")
    p.add_argument("--extra", nargs="*", default=[])
    args = p.parse_args()

    nranks = int(os.environ.get("OMPI_COMM_WORLD_SIZE",
                  os.environ.get("PMI_SIZE", "1")))

    total_lps = args.num_clients + 1
    if total_lps % nranks != 0:
        raise SystemExit(
            f"num_clients+1 ({total_lps}) must divide evenly into {nranks} MPI ranks "
            f"(LINEAR mapping). Try --num-clients={(total_lps // nranks) * nranks - 1}.")
    lps_per_rank = total_lps // nranks

    global NUM_CLIENTS, TOTAL_ROUNDS, MAX_CONCURRENCY
    global COMPUTE_MEAN_BASE, MODEL_BYTES
    NUM_CLIENTS = args.num_clients
    TOTAL_ROUNDS = args.rounds
    MAX_CONCURRENCY = min(args.concurrency, args.num_clients)
    COMPUTE_MEAN_BASE = args.compute_mean
    MODEL_BYTES = args.model_bytes

    sim = ross.Simulator(
        lps_per_rank=lps_per_rank,
        type_map=lambda gid: "server" if gid == 0 else "client",
        synch=args.synch,
        end_time=args.end,
        max_msg_size=64,           # Pull/Done pickle to ~30 B
        extra_args=args.extra,
    )
    sim.run()


if __name__ == "__main__":
    main()
