"""Airport model in py-ross.

Towers exchange Arrive and Land events. Events are dispatched by type using
the pickled `payload` (an `Arrive` or `Land` dataclass) rather than a
numeric tag.

The `sender` argument on every handler is the gid of the LP that sent the
event. We don't strictly need it here, but a "reply to sender" pattern looks
like:

    def on_event(self, sender, msg, now):
        if isinstance(msg.payload, Ping):
            self.send(sender, 0.1, payload=Pong())   # reply path
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import ross

TOTAL_TOWERS: int = 0
MEAN_FLIGHT: float = 5.0
MEAN_LAND: float = 1.0
LOOKAHEAD: float = 0.5


# Payload classes must be defined at module scope so they're importable on
# every rank for cross-rank pickle round-trips.
@dataclass
class Arrive:
    pass


@dataclass
class Land:
    pass


@ross.lp("tower")
class Tower(ross.LP):
    in_air: int
    landed: int
    dispatched: int

    def init(self) -> None:
        for _ in range(2):
            d = self.rand_exponential(MEAN_FLIGHT) + LOOKAHEAD
            self.send(self.gid, d, payload=Arrive())
        self.in_air = 0
        self.landed = 0
        self.dispatched = 0

    def on_event(self, sender: int, msg: ross.Msg, now: float) -> None:
        p = msg.payload
        if isinstance(p, Arrive):
            self.in_air += 1
            d = self.rand_exponential(MEAN_LAND) + LOOKAHEAD + 0.1 * self.in_air
            self.send(self.gid, d, payload=Land())
        elif isinstance(p, Land):
            self.in_air = max(0, self.in_air - 1)
            self.landed += 1
            dest = int(self.rand_integer(0, TOTAL_TOWERS - 1))
            d = self.rand_exponential(MEAN_FLIGHT) + LOOKAHEAD
            self.send(dest, d, payload=Arrive())
            self.dispatched += 1

    def reverse_event(self, sender: int, msg: ross.Msg, bf: ross.BitField) -> None:
        p = msg.payload
        if isinstance(p, Arrive):
            self.rev_rand_exponential()
            self.in_air -= 1
        elif isinstance(p, Land):
            self.dispatched -= 1
            self.rev_rand_exponential()
            self.rev_rand_integer()
            self.landed -= 1
            self.in_air += 1


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--towers-per-rank", type=int, default=4)
    p.add_argument("--synch", default="sequential",
                   choices=["sequential", "conservative",
                            "optimistic", "rollback_check"])
    p.add_argument("--end", type=float, default=100.0)
    p.add_argument("--nkp", type=int, default=16)
    p.add_argument("--extra", nargs="*", default=[])
    args = p.parse_args()

    nranks = int(os.environ.get("OMPI_COMM_WORLD_SIZE",
                  os.environ.get("PMI_SIZE", "1")))

    global TOTAL_TOWERS
    TOTAL_TOWERS = args.towers_per_rank * nranks

    sim = ross.Simulator(
        lps_per_rank=args.towers_per_rank,
        type_map=lambda gid: "tower",
        synch=args.synch,
        end_time=args.end,
        nkp=args.nkp,
        max_msg_size=256,          # Arrive/Land pickle to ~30 B; 256 is ample
        extra_args=args.extra,
    )
    sim.run()


if __name__ == "__main__":
    main()
