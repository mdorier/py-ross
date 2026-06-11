"""PHOLD ported to py-ross.

Each LP fires `start_events` self-events at start, then on each event
forwards a new event either to a random remote LP (with probability
`percent_remote`) or back to itself, with an exponentially-distributed
delay plus lookahead.

Events carry no payload — the arrival itself is the signal.
"""

from __future__ import annotations

import argparse
import os

import ross

TOTAL_LPS: int = 0
MEAN: float = 1.0
LOOKAHEAD: float = 1.0
PERCENT_REMOTE: float = 0.25
START_EVENTS: int = 1


@ross.lp("phold")
class PHOLD(ross.LP):
    def init(self) -> None:
        for _ in range(START_EVENTS):
            delay = self.rand_exponential(MEAN) + LOOKAHEAD
            self.send(self.gid, delay)

    def on_event(self, sender: int, msg: ross.Msg, now: float) -> None:
        u = self.rand_uniform()
        if u <= PERCENT_REMOTE:
            dest = self.rand_integer(0, TOTAL_LPS - 1)
        else:
            dest = self.gid
        delay = self.rand_exponential(MEAN) + LOOKAHEAD
        self.send(int(dest), delay)

    def reverse_event(self, sender: int, msg: ross.Msg, bf: ross.BitField) -> None:
        self.rev_rand_exponential()
        self.rev_rand_integer()
        self.rev_rand_uniform()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--lps-per-rank", type=int, default=8)
    p.add_argument("--synch", default="sequential",
                   choices=["sequential", "conservative",
                            "optimistic", "rollback_check"])
    p.add_argument("--end", type=float, default=100.0)
    p.add_argument("--mean", type=float, default=1.0)
    p.add_argument("--lookahead", type=float, default=1.0)
    p.add_argument("--remote", type=float, default=0.25)
    p.add_argument("--start-events", type=int, default=1)
    p.add_argument("--nkp", type=int, default=16)
    p.add_argument("--extra", nargs="*", default=[])
    args = p.parse_args()

    global TOTAL_LPS, MEAN, LOOKAHEAD, PERCENT_REMOTE, START_EVENTS
    MEAN = args.mean - args.lookahead
    LOOKAHEAD = args.lookahead
    PERCENT_REMOTE = args.remote
    START_EVENTS = args.start_events

    nranks = int(os.environ.get("OMPI_COMM_WORLD_SIZE",
                  os.environ.get("PMI_SIZE", "1")))
    TOTAL_LPS = args.lps_per_rank * nranks

    sim = ross.Simulator(
        lps_per_rank=args.lps_per_rank,
        type_map=lambda gid: "phold",
        synch=args.synch,
        end_time=args.end,
        nkp=args.nkp,
        extra_args=args.extra,
    )
    sim.run()


if __name__ == "__main__":
    main()
