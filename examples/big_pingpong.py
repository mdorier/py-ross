"""2-LP ping-pong carrying a configurable-size bytes payload.

Validates that the MPI on-wire path correctly transports `max_msg_size`-sized
events. With default --payload-bytes=32768, this round-trips a 32 KiB chunk
between ranks several times.
"""
from __future__ import annotations

import argparse
import os

import ross

BYTES: int = 0
TARGET: int = 5


@ross.lp("big")
class Big(ross.LP):
    received: int

    def init(self) -> None:
        self.received = 0
        if self.gid == 0:
            self.send(1, 1.0, payload=b"\x42" * BYTES)

    def on_event(self, sender: int, msg: ross.Msg, now: float, bf: ross.BitField) -> None:
        data = msg.payload
        assert isinstance(data, (bytes, bytearray)), f"unexpected payload {type(data)}"
        assert len(data) == BYTES, f"size mismatch: got {len(data)}, want {BYTES}"
        # Cheap content check: spot-check first/last byte
        assert data[0] == 0x42 and data[-1] == 0x42
        self.received += 1
        if self.received < TARGET:
            self.send(sender, 1.0, payload=b"\x42" * BYTES)

    def final(self) -> None:
        print(f"LP {self.gid} received {self.received} {BYTES}-byte payloads")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--payload-bytes", type=int, default=32 * 1024)
    p.add_argument("--target", type=int, default=5)
    p.add_argument("--synch", default="conservative",
                   choices=["sequential", "conservative", "optimistic"])
    p.add_argument("--end", type=float, default=100.0)
    p.add_argument("--extra", nargs="*", default=[])
    args = p.parse_args()

    global BYTES, TARGET
    BYTES = args.payload_bytes
    TARGET = args.target

    nranks = int(os.environ.get("OMPI_COMM_WORLD_SIZE",
                  os.environ.get("PMI_SIZE", "1")))
    if nranks not in (1, 2):
        raise SystemExit("big_pingpong runs on 1 or 2 ranks (2 LPs total)")

    # Need headroom for pickle framing (~10-15 B for bytes objects this size).
    sim = ross.Simulator(
        lps_per_rank=2 // nranks,
        type_map=lambda gid: "big",
        synch=args.synch,
        end_time=args.end,
        max_msg_size=BYTES + 256,
        extra_args=args.extra,
    )
    sim.run()


if __name__ == "__main__":
    main()
