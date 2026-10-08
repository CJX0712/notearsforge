"""Determinism gates: identical seed -> identical output, byte-for-byte.

These are hard SOP gates: a causal-discovery system that is not reproducible
cannot be trusted, so we assert it directly here.
"""

from __future__ import annotations

import numpy as np

from causal.greedy import GES
from causal.notears_linear import NotearsLinear
from causal.pc import PC
from core.seed import set_all
from data.generators import make_dataset


def _small_dataset(seed=7, d=10, n=600):
    set_all(seed)
    return make_dataset(d=d, n=n, graph_type="er", sem="lin", seed=seed, er_p=2)


def test_ges_adjacency_deterministic():
    ds = _small_dataset()
    a1 = GES().fit(ds.X).adjacency()
    a2 = GES().fit(ds.X).adjacency()
    assert bool((a1 == a2).all())


def test_pc_adjacency_deterministic():
    ds = _small_dataset()
    a1 = PC().fit(ds.X).adjacency()
    a2 = PC().fit(ds.X).adjacency()
    assert bool((a1 == a2).all())


def test_notears_adjacency_deterministic():
    ds = _small_dataset(d=8, n=400)
    a1 = (
        NotearsLinear(lambda1=0.05, max_iter=30, inner_iter=200, threshold=0.2)
        .fit(ds.X)
        .adjacency()
    )
    a2 = (
        NotearsLinear(lambda1=0.05, max_iter=30, inner_iter=200, threshold=0.2)
        .fit(ds.X)
        .adjacency()
    )
    assert bool((a1 == a2).all())


def test_make_dataset_deterministic():
    ds1 = make_dataset(d=12, n=500, graph_type="er", sem="lin", seed=11)
    ds2 = make_dataset(d=12, n=500, graph_type="er", sem="lin", seed=11)
    assert np.array_equal(ds1.X, ds2.X)
    assert np.array_equal(ds1.adj_true, ds2.adj_true)


def test_ges_vs_pc_baseline_on_headline_regime():
    """Headline gate: GES must beat PC on linear-Gaussian ER (the regime where
    the score-based flagship is provably ahead). Uses the SOP significance rule
    mean-diff > 1/2 (sigma_ges + sigma_pc) over >=3 seeds."""
    from pipeline.benchmark import DEFAULT_METHODS, run_benchmark

    # Headline regime: linear-Gaussian Erdos-Renyi, sparse (er_p=1) -- the
    # regime where the score-based flagship (GES) is provably ahead of the
    # constraint-based baseline (PC). Matches examples/run_demo.py config.
    seeds = (7, 11, 23, 41, 53)
    suite = [make_dataset(d=12, n=1000, graph_type="er", sem="lin", seed=s, er_p=1) for s in seeds]
    records, summary = run_benchmark(suite=suite, methods=DEFAULT_METHODS)

    g = summary["ges"]
    p = summary["pc"]
    assert not g["skipped"] and not p["skipped"]
    delta = g["skel_f1_mean"] - p["skel_f1_mean"]
    half_sum_sigma = 0.5 * (g["skel_f1_std"] + p["skel_f1_std"])
    assert delta > half_sum_sigma, (
        f"GES vs PC not significantly ahead: delta={delta:.4f} need > {half_sum_sigma:.4f}"
    )
