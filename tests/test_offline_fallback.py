"""Offline-fallback gate: a missing backend must be *skipped*, never faked.

The benchmark wraps every method factory in try/except; if a backend raises we
record a skipped result and continue. This is the SOP "do not fabricate numbers"
rule, and the skip path must itself be tested.
"""

from __future__ import annotations

from core.errors import BackendUnavailable
from core.seed import set_all
from data.generators import make_dataset
from pipeline.benchmark import run_benchmark


def _raising_factory():
    # simulates an optional backend (e.g. causallearn) that failed to import
    def _make():
        raise BackendUnavailable("causallearn not installed (simulated)")

    return _make


def test_offline_fallback_skips_missing_backend():
    set_all(7)
    ds = make_dataset(d=10, n=400, graph_type="er", sem="lin", seed=7)
    suite = [ds]
    methods = {"broken": (_raising_factory(), "simulated missing backend")}
    records, summary = run_benchmark(suite=suite, methods=methods)
    assert len(records) == 1
    assert records[0].skipped is True
    assert "causallearn" in records[0].note
    assert summary["broken"]["skipped"] is True
