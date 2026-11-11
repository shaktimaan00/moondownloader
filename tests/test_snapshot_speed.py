"""Test: `Engine.snapshot()` publishes its speed as decimal MB/s.

Regression test for #84. `snapshot()` divided the aggregate byte rate by
1_048_576 while every consumer labels it "MB/s": `web/app.js` prints it through
`fmtSpeed` and `moon_cli.py` through `_fmt_speed`, both of which scale down to
KB/s by 1024. The per-file rates beside it were already divided by 1e6, so the
aggregate line read 4.9% lower than the files it was summarising.

Nothing here sleeps. The sample timestamps are placed relative to a single
`time.monotonic()` reading and every expectation is derived from the rule

    speed_mbs = bytes in the trailing 3.0 s / min(3.0, elapsed) / 1e6
    eta_s     = files remaining * bytes per finished file / (speed_mbs * 1e6)
"""
from __future__ import annotations

import ast
import pathlib
import time

import pytest

import moon_engine
from moon_engine import Engine

MB = 1_000_000


def _engine(now: float, t0: float, sizes=(MB, MB, MB)) -> Engine:
    """An engine that has received `sizes` bytes over the last 0.2 s."""
    e = Engine()
    e._bytes_acc.extend((now - 0.1 * (len(sizes) - 1 - i), size)
                        for i, size in enumerate(sizes))
    e._t0 = now - t0
    return e


def test_snapshot_speed_is_the_window_average_in_decimal_mb():
    # 3 MB inside the trailing window, the span clamped to 3.0 s because the
    # run started long ago, so 3 MB / 3.0 s = 1.0 MB/s. Neither the peak of a
    # single sample (10 MB/s) nor the MiB reading of the same bytes (0.954) is
    # this number.
    now = time.monotonic()
    got = _engine(now, t0=1000.0).snapshot()["metrics"]["speed_mbs"]
    assert got == pytest.approx(1.0)


def test_snapshot_speed_uses_elapsed_time_until_the_window_fills():
    # 1.5 s into the run the span is the run's own age, not the 3.0 s window:
    # 3 MB / 1.5 s = 2.0 MB/s.
    now = time.monotonic()
    got = _engine(now, t0=1.5).snapshot()["metrics"]["speed_mbs"]
    assert got == pytest.approx(2.0)


def test_snapshot_eta_reads_its_speed_back_at_the_same_1e6():
    # 3 MB per finished file, two still queued, 1.0 MB/s -> 6 s. The ETA turns
    # the published figure back into bytes, so it reads 6 s under either
    # divisor; what it catches is the divisor being changed on one side only.
    now = time.monotonic()
    e = _engine(now, t0=1000.0)
    e._dl_done, e._dl_total = 1, 3

    metrics = e.snapshot()["metrics"]

    assert metrics["speed_mbs"] == pytest.approx(1.0)
    assert metrics["eta_s"] == pytest.approx(6.0)


def test_no_mib_divisor_is_left_in_the_engine():
    """Guard the convention, so a second unit cannot creep back in unnoticed.

    Scanned from the syntax tree rather than the text, so the comment above can
    name the old divisor while this still fails if one is divided by again.
    """
    tree = ast.parse(pathlib.Path(moon_engine.__file__).read_text(encoding="utf-8"))
    divisors = {
        node.right.value
        for node in ast.walk(tree)
        if isinstance(node, ast.BinOp)
        and isinstance(node.op, ast.Div)
        and isinstance(node.right, ast.Constant)
        and isinstance(node.right.value, (int, float))
        and not isinstance(node.right.value, bool)
        and node.right.value in (1_048_576, 1024 * 1024, 2 ** 20)
    }

    assert not divisors, (
        f"the engine divides by {sorted(divisors)} again; every rate it "
        "publishes is read as MB/s by both front-ends")
