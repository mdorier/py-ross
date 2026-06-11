from typing import Callable, TypeVar

from ._ross import LP, BitField, Msg, Simulator, register_lp_type

T = TypeVar("T", bound=type)

def lp(name: str) -> Callable[[T], T]: ...

__all__ = ["LP", "Msg", "BitField", "Simulator", "register_lp_type", "lp"]
