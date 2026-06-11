from __future__ import annotations

import inspect
from typing import Callable, TypeVar

from ._ross import LP, Msg, BitField, Simulator, register_lp_type

# Internal alias kept for clarity in the decorator.
_LP = LP

T = TypeVar("T", bound=type)

# Names the LP runtime provides on every instance. Decorated classes
# must not define attributes with these names (the LP base would silently
# shadow them, which is almost never what the user wants).
_RESERVED = frozenset({
    "gid", "now", "send",
    "rand_uniform", "rand_exponential", "rand_integer",
    "rev_rand_uniform", "rev_rand_exponential", "rev_rand_integer",
})

# Names of the lifecycle hooks the LP runtime will look for. A decorated
# class is expected to provide at least `on_event`. The others are optional.
_LIFECYCLE = frozenset({
    "init", "pre_run", "on_event", "reverse_event", "commit_event", "final",
})


def lp(name: str) -> Callable[[T], T]:
    """Decorator: register a plain Python class as an LP type.

    Usage:
        @ross.lp("server")
        class FedAvgServer:
            def init(self): ...
            def on_event(self, msg, now): ...

    The decorated class does NOT need to inherit from anything at runtime —
    the decorator synthesizes an internal LP subclass and registers it. The
    original class is returned unchanged.

    To get type-checker support for the runtime-provided methods
    (`self.gid`, `self.now`, `self.send`, `self.rand_*`, `self.rev_rand_*`),
    explicitly subclass `ross.LP`:

        @ross.lp("server")
        class FedAvgServer(ross.LP):
            def init(self) -> None: ...
            def on_event(self, msg: ross.Msg, now: float) -> None: ...

    For optimistic mode, also define
    `reverse_event(self, msg: ross.Msg, bf: ross.BitField) -> None`.
    """
    if not isinstance(name, str) or not name:
        raise TypeError("@ross.lp(name): name must be a non-empty string")

    def decorate(cls: T) -> T:
        if not inspect.isclass(cls):
            raise TypeError("@ross.lp can only decorate classes")

        own = set(vars(cls))

        # Reject classes that shadow LP-provided names.
        conflicts = _RESERVED & own
        if conflicts:
            raise TypeError(
                f"@ross.lp('{name}'): class {cls.__name__} defines attributes "
                f"that conflict with the LP runtime API: {sorted(conflicts)}. "
                f"Rename them — these are provided by the LP base."
            )

        # Require at least one lifecycle hook. on_event is the meaningful one;
        # an LP with no on_event would never react to anything.
        if "on_event" not in own and not any(
            hasattr(cls, h) and getattr(cls, h) is not getattr(object, h, None)
            for h in _LIFECYCLE
        ):
            raise TypeError(
                f"@ross.lp('{name}'): class {cls.__name__} must define at least "
                f"one of {sorted(_LIFECYCLE)} (typically on_event)."
            )

        # If the user defined __init__, it must be callable with no args.
        if "__init__" in own:
            try:
                sig = inspect.signature(cls.__init__)
            except (TypeError, ValueError):
                sig = None  # builtin / can't introspect; defer to runtime
            if sig is not None:
                params = [p for p in sig.parameters.values() if p.name != "self"]
                required = [p for p in params
                            if p.default is inspect.Parameter.empty
                            and p.kind in (inspect.Parameter.POSITIONAL_ONLY,
                                           inspect.Parameter.POSITIONAL_OR_KEYWORD,
                                           inspect.Parameter.KEYWORD_ONLY)]
                if required:
                    raise TypeError(
                        f"@ross.lp('{name}'): {cls.__name__}.__init__ has required "
                        f"parameters {[p.name for p in required]}; the LP runtime "
                        f"instantiates each LP with no arguments. Move setup into "
                        f"`init(self)` instead."
                    )

        # Synthesize a fresh subclass of the C++ LP base, copying the user's
        # methods/attributes onto it. We can't multi-inherit (cls, _LP) because
        # nanobind's metaclass requires its base to be the sole base. Copying
        # is fine: a single LP class is instantiated per LP, not the user's
        # class directly. The user's class is returned unchanged.
        attrs = {"__module__": cls.__module__,
                 "__qualname__": cls.__qualname__,
                 "__doc__": cls.__doc__}
        for k, v in vars(cls).items():
            if k in ("__dict__", "__weakref__"):
                continue
            attrs[k] = v
        combined = type(cls.__name__, (_LP,), attrs)
        register_lp_type(name, combined)
        return cls

    return decorate


__all__ = ["LP", "Msg", "BitField", "Simulator", "register_lp_type", "lp"]
