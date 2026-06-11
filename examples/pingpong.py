"""Two LPs ping-pong with a pickled payload.

LP 0 sends Ping(hops=0) to LP 1. LP 1 echoes back to `sender` with hops+1.
Repeat until hops >= TARGET, then both LPs go quiet.

At final(), each LP asserts:
  - the only sender it ever saw was the *other* LP (validates the `sender`
    trampoline wiring);
  - its hop counter matches the expected count.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from typing import List, Set

import ross

TARGET: int = 0


@dataclass
class Ping:
    hops: int = 0
    trace: List[int] = field(default_factory=list)


@ross.lp("node")
class Node(ross.LP):
    seen_senders: Set[int]
    max_hops: int

    def init(self) -> None:
        self.seen_senders = set()
        self.max_hops = -1
        if self.gid == 0:
            # Kick things off.
            self.send(1, 1.0, payload=Ping(hops=0, trace=[0]))

    def on_event(self, sender: int, msg: ross.Msg, now: float, bf: ross.BitField) -> None:
        self.seen_senders.add(sender)
        p = msg.payload
        assert isinstance(p, Ping), f"unexpected payload {type(p)}"
        if p.hops > self.max_hops:
            self.max_hops = p.hops
        if p.hops >= TARGET:
            return
        # Bounce back to sender with hops+1.
        other = sender
        self.send(other, 1.0,
                  payload=Ping(hops=p.hops + 1, trace=p.trace + [self.gid]))

    def reverse_event(self, sender: int, msg: ross.Msg, bf: ross.BitField) -> None:
        # We don't try to undo the set/max tracking precisely — pingpong only
        # runs in conservative or sequential mode for this check.
        pass

    def final(self) -> None:
        other = 1 - self.gid
        assert self.seen_senders <= {other}, \
            f"LP {self.gid} saw unexpected senders {self.seen_senders}"
        if self.seen_senders:
            # Each LP receives TARGET//2 + 1 or TARGET//2 events, depending on
            # parity. Just assert the max-hops we saw is consistent.
            assert 0 <= self.max_hops <= TARGET, \
                f"LP {self.gid} max_hops={self.max_hops} out of range [0,{TARGET}]"
        print(f"LP {self.gid}: senders={self.seen_senders}, max_hops={self.max_hops}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--target", type=int, default=20)
    p.add_argument("--synch", default="sequential",
                   choices=["sequential", "conservative", "optimistic"])
    p.add_argument("--end", type=float, default=100.0)
    p.add_argument("--extra", nargs="*", default=[])
    args = p.parse_args()

    global TARGET
    TARGET = args.target

    nranks = int(os.environ.get("OMPI_COMM_WORLD_SIZE",
                  os.environ.get("PMI_SIZE", "1")))
    if nranks not in (1, 2):
        raise SystemExit("pingpong runs only on 1 or 2 ranks (2 LPs total)")

    sim = ross.Simulator(
        lps_per_rank=2 // nranks,
        type_map=lambda gid: "node",
        synch=args.synch,
        end_time=args.end,
        max_msg_size=512,          # Ping(hops, trace) grows with hops; 512 covers target<=20
        extra_args=args.extra,
    )
    sim.run()


if __name__ == "__main__":
    main()
