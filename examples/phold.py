"""PHOLD ported to py-ross.

Each LP fires `start_events` self-events at start, then on each event
forwards a new event either to a random remote LP (with probability
`percent_remote`) or back to itself, with an exponentially-distributed
delay plus lookahead.

This mirrors ROSS/models/phold/phold.main.c closely.
"""

import argparse
import ross

# Globals shared across all LPs (set in main before run()).
TOTAL_LPS = 0
MEAN = 1.0
LOOKAHEAD = 1.0
PERCENT_REMOTE = 0.25
START_EVENTS = 1


class PHOLD(ross.LP):
    def init(self):
        # Bootstrap: schedule `START_EVENTS` self-events.
        for _ in range(START_EVENTS):
            delay = self.rand_exponential(MEAN) + LOOKAHEAD
            self.send(self.gid, delay, msg_type=0)

    def on_event(self, msg, now):
        # Decide remote vs local.
        u = self.rand_uniform()
        if u <= PERCENT_REMOTE:
            dest = self.rand_integer(0, TOTAL_LPS - 1)
        else:
            dest = self.gid
        delay = self.rand_exponential(MEAN) + LOOKAHEAD
        self.send(int(dest), delay, msg_type=0)

    def reverse_event(self, msg, bf):
        # Undo, in reverse order, the three RNG draws above.
        self.rev_rand_exponential()
        self.rev_rand_integer()  # the rand_integer call (or unused branch)
        self.rev_rand_uniform()

    def final(self):
        pass


def main():
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
    p.add_argument("--extra", nargs="*", default=[],
                   help="Extra args passed through to tw_init (e.g. --extramem=...)")
    args = p.parse_args()

    global TOTAL_LPS, MEAN, LOOKAHEAD, PERCENT_REMOTE, START_EVENTS
    MEAN = args.mean - args.lookahead  # match phold.main.c semantics
    LOOKAHEAD = args.lookahead
    PERCENT_REMOTE = args.remote
    START_EVENTS = args.start_events

    # We don't know nranks yet, but tw_init populates it before c_init runs.
    # We need TOTAL_LPS for rand_integer(0, total-1). Use a sentinel and
    # patch it on the first event — or simpler: compute it from extras.
    # The simplest fix: pass lps-per-rank * mpirun_np through env or
    # accept that single-rank means TOTAL_LPS == lps_per_rank.
    # For v0 we set TOTAL_LPS = lps_per_rank * 1 and override via --extra
    # if running multi-rank. mpirun-aware sizing:
    import os
    np_env = int(os.environ.get("OMPI_COMM_WORLD_SIZE",
                  os.environ.get("PMI_SIZE",
                  os.environ.get("MPI_LOCALNRANKS", "1"))))
    TOTAL_LPS = args.lps_per_rank * np_env

    sim = ross.Simulator(
        lps_per_rank=args.lps_per_rank,
        type_map=lambda gid: "phold",
        synch=args.synch,
        end_time=args.end,
        nkp=args.nkp,
        extra_args=args.extra,
    )
    ross.register_lp_type("phold", PHOLD)
    sim.run()


if __name__ == "__main__":
    main()
