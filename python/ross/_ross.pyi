"""Type stubs for the compiled `ross._ross` extension."""

from typing import Any, Callable, Sequence

DEFAULT_MAX_MSG_SIZE: int
"""Default value of `Simulator.max_msg_size` when none is supplied (256)."""

class Msg:
    """Event payload view. Use `.payload` to get the unpickled object (or
    `None` for an empty event). The sender's LP gid is delivered as a separate
    `sender` argument to event handlers, not as a field on Msg."""
    @property
    def payload(self) -> Any:
        """The unpickled Python object the sender attached, or None if the
        event was sent with payload=None."""
        ...

class BitField:
    """View onto ROSS's per-event `tw_bf` (32 single-bit flags `c0`..`c31`)."""
    c0: bool;  c1: bool;  c2: bool;  c3: bool
    c4: bool;  c5: bool;  c6: bool;  c7: bool
    c8: bool;  c9: bool;  c10: bool; c11: bool
    c12: bool; c13: bool; c14: bool; c15: bool
    c16: bool; c17: bool; c18: bool; c19: bool
    c20: bool; c21: bool; c22: bool; c23: bool
    c24: bool; c25: bool; c26: bool; c27: bool
    c28: bool; c29: bool; c30: bool; c31: bool

class LP:
    """Base class for ROSS LPs. End users should not subclass directly;
    use `@ross.lp("name")` on a plain class instead."""

    def __init__(self) -> None: ...

    # Lifecycle hooks — override on subclasses.
    def init(self) -> None: ...
    def pre_run(self) -> None: ...
    def on_event(self, sender: int, msg: Msg, now: float) -> None: ...
    def reverse_event(self, sender: int, msg: Msg, bf: BitField) -> None: ...
    def commit_event(self, sender: int, msg: Msg) -> None: ...
    def final(self) -> None: ...

    # Runtime-provided attributes / methods.
    @property
    def gid(self) -> int:
        """The global LP id assigned by ROSS."""
        ...
    @property
    def now(self) -> float:
        """Current virtual time on this LP."""
        ...

    def rand_uniform(self) -> float: ...
    def rand_exponential(self, mean: float) -> float: ...
    def rand_integer(self, lo: int, hi: int) -> int: ...

    def rev_rand_uniform(self) -> None: ...
    def rev_rand_exponential(self) -> None: ...
    def rev_rand_integer(self) -> None: ...

    def send(
        self,
        dest_gid: int,
        ts_offset: float,
        payload: Any = None,
    ) -> None:
        """Schedule an event at `dest_gid` to arrive `ts_offset` ahead.

        If `payload` is given (any pickleable object), it is pickled into the
        event buffer. The class of the payload must be importable on every
        rank for cross-rank sends. Raises `OverflowError` if the pickled
        payload exceeds the active Simulator's `max_msg_size`."""
        ...


class Simulator:
    max_msg_size: int  # read-only after construction
    def __init__(
        self,
        lps_per_rank: int,
        type_map: Callable[[int], str],
        synch: str = "conservative",
        end_time: float = 100.0,
        nkp: int = 16,
        max_msg_size: int = 256,
        extra_args: Sequence[str] = (),
    ) -> None: ...
    def run(self) -> None: ...


def register_lp_type(name: str, cls: type) -> None:
    """Register a Python class as the LP type identified by `name`.

    Prefer the `@ross.lp(name)` decorator over calling this directly."""
    ...
