"""Child process: a 1-LP simulator that tries to send an over-large payload.

Prints "OK" on stdout if OverflowError was raised as expected.
"""
from __future__ import annotations

import sys
import ross

MAX_MSG_SIZE = 64


@ross.lp("big")
class BigSender(ross.LP):
    def init(self) -> None:
        # Build a payload guaranteed to exceed the active max_msg_size.
        big = b"x" * (MAX_MSG_SIZE * 4)
        try:
            self.send(self.gid, 1.0, payload=big)
        except OverflowError as e:
            msg = str(e)
            assert "max_msg_size" in msg, f"limit name missing: {msg}"
            assert str(MAX_MSG_SIZE) in msg, f"limit value missing: {msg}"
            assert str(len(big)) in msg or "bytes" in msg, \
                f"actual size not mentioned: {msg}"
            print("OK", flush=True)
            sys.exit(0)
        print("FAIL: no exception raised", flush=True)
        sys.exit(1)

    def on_event(self, sender: int, msg: ross.Msg, now: float, bf: ross.BitField) -> None:
        pass


def main() -> None:
    sim = ross.Simulator(
        lps_per_rank=1,
        type_map=lambda gid: "big",
        synch="sequential",
        end_time=10.0,
        max_msg_size=MAX_MSG_SIZE,
    )
    sim.run()


if __name__ == "__main__":
    main()
