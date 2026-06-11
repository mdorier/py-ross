"""Airport model in py-ross.

Towers exchange ARRIVE and LAND events. Each Tower has a single runway
of fixed in-air capacity. When a plane arrives at a tower:
  - If the runway is free, schedule LAND `landing_time` ahead.
  - Else, increment the in-air queue and bump the schedule.
On LAND, the tower frees up the runway and dispatches the plane to a
random other tower with `flight_time` delay.

Each LP is a Tower. Planes have no LP; they are implicit in messages.
"""

import argparse
import ross

TOTAL_TOWERS = 0
MEAN_FLIGHT = 5.0
MEAN_LAND = 1.0
LOOKAHEAD = 0.5

ARRIVE = 1
LAND = 2


class Tower(ross.LP):
    def init(self):
        # Each tower bootstraps a few planes.
        for _ in range(2):
            d = self.rand_exponential(MEAN_FLIGHT) + LOOKAHEAD
            self.send(self.gid, d, msg_type=ARRIVE)
        # Per-LP state lives on `self` (invisible to ROSS).
        self.in_air = 0
        self.landed = 0
        self.dispatched = 0

    def on_event(self, msg, now):
        if msg.msg_type == ARRIVE:
            self.in_air += 1
            # Schedule LAND after a stack delay proportional to queue depth.
            d = self.rand_exponential(MEAN_LAND) + LOOKAHEAD + 0.1 * self.in_air
            self.send(self.gid, d, msg_type=LAND)
        elif msg.msg_type == LAND:
            self.in_air = max(0, self.in_air - 1)
            self.landed += 1
            # Dispatch to a random other tower.
            dest = int(self.rand_integer(0, TOTAL_TOWERS - 1))
            d = self.rand_exponential(MEAN_FLIGHT) + LOOKAHEAD
            self.send(dest, d, msg_type=ARRIVE)
            self.dispatched += 1

    def reverse_event(self, msg, bf):
        if msg.msg_type == ARRIVE:
            # undo: rand_exponential, in_air increment
            self.rev_rand_exponential()
            self.in_air -= 1
        elif msg.msg_type == LAND:
            # undo in reverse order: rand_exponential (dispatch delay),
            # rand_integer (dest), state mutations.
            self.dispatched -= 1
            self.rev_rand_exponential()
            self.rev_rand_integer()
            self.landed -= 1
            self.in_air += 1

    def final(self):
        # Sequential / single-rank only: print local stats.
        pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--towers-per-rank", type=int, default=4)
    p.add_argument("--synch", default="sequential",
                   choices=["sequential", "conservative",
                            "optimistic", "rollback_check"])
    p.add_argument("--end", type=float, default=100.0)
    p.add_argument("--nkp", type=int, default=16)
    p.add_argument("--extra", nargs="*", default=[])
    args = p.parse_args()

    import os
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
        extra_args=args.extra,
    )
    ross.register_lp_type("tower", Tower)
    sim.run()


if __name__ == "__main__":
    main()
