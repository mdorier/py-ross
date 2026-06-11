"""Standalone smoke tests for the new payload API.

Run with:
    PYTHONPATH=python python tests/test_payload.py
"""
from __future__ import annotations

import sys
import subprocess
from pathlib import Path

import ross


def test_constants() -> None:
    assert isinstance(ross.DEFAULT_MAX_MSG_SIZE, int)
    assert ross.DEFAULT_MAX_MSG_SIZE > 0
    print(f"  DEFAULT_MAX_MSG_SIZE = {ross.DEFAULT_MAX_MSG_SIZE}")


def test_simulator_max_msg_size() -> None:
    sim = ross.Simulator(lps_per_rank=1, type_map=lambda g: "x")
    assert sim.max_msg_size == ross.DEFAULT_MAX_MSG_SIZE
    sim2 = ross.Simulator(lps_per_rank=1, type_map=lambda g: "x", max_msg_size=128)
    assert sim2.max_msg_size == 128
    raised = False
    try:
        ross.Simulator(lps_per_rank=1, type_map=lambda g: "x", max_msg_size=0)
    except (RuntimeError, ValueError):
        raised = True
    assert raised, "max_msg_size=0 should be rejected"


def test_overflow_raises() -> None:
    """An over-large payload must raise OverflowError at send() time.

    We can't easily test this without running a Simulator. Instead, we
    spawn a child that runs a 1-LP simulator and tries to send a too-big
    payload from init().
    """
    here = Path(__file__).resolve().parent.parent
    script = here / "tests" / "_overflow_child.py"
    env = {"PYTHONPATH": str(here / "python"), "PATH": "/usr/bin:/bin"}
    res = subprocess.run([sys.executable, str(script)],
                         env=env, capture_output=True, text=True)
    # The child writes "OK" on stdout when OverflowError was raised as
    # expected; anything else is a failure.
    assert "OK" in res.stdout, (
        f"overflow child did not report OK\n"
        f"  stdout: {res.stdout!r}\n  stderr: {res.stderr!r}"
    )


def main() -> None:
    failures = []
    for fn in [test_constants, test_simulator_max_msg_size, test_overflow_raises]:
        print(f"... {fn.__name__}")
        try:
            fn()
        except AssertionError as e:
            failures.append((fn.__name__, str(e)))
            print(f"    FAIL: {e}")
        else:
            print(f"    ok")
    if failures:
        print(f"\n{len(failures)} failure(s)")
        sys.exit(1)
    print("\nall good")


if __name__ == "__main__":
    main()
