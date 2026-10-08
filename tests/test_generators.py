"""Data-generator regression: the DGP must carry real causal signal.

The single most damaging bug we hit was ``_er_dag`` using ``tril`` instead of
``triu``: parents then had a *higher* index than the child, but the linear SEM
evaluates nodes in index order, so each parent was still 0 (uninitialised) when
its child was generated -- the data became pure noise and *every* algorithm
recovered the empty graph (skelF1 = 0). This test locks the fix.
"""

from __future__ import annotations

import numpy as np

from causal.greedy import GES
from data.generators import _er_dag, make_dataset
from eval.metrics import evaluate


def test_er_dag_is_upper_triangular_topological_order():
    rng = np.random.default_rng(3)
    B = _er_dag(d=12, p=2.0, rng=rng)
    # no edge from a higher-index node to a lower-index node -> topological order
    lower = np.tril(B, k=-1)
    assert int(lower.sum()) == 0, "ER DAG must be upper-triangular (parent idx < child idx)"
    assert B.shape == (12, 12)


def test_generated_data_carries_causal_signal():
    """Regression for the tril/triu bug: on a small ER+linear dataset, GES must
    recover the skeleton well above chance. tril bug -> skelF1 == 0."""
    np.random.seed(7)
    ds = make_dataset(d=10, n=800, graph_type="er", sem="lin", seed=7, er_p=2)
    adj = GES().fit(ds.X).adjacency()
    res = evaluate(ds.adj_true, adj)
    assert res["skel_f1"] > 0.5, (
        f"DGP lost causal signal (skelF1={res['skel_f1']:.3f}); check _er_dag uses triu, not tril"
    )


def test_make_dataset_permuted_adjacency_consistency():
    # permutation preserves symmetry of the edge *set* (skeleton) up to the perm
    ds = make_dataset(d=12, n=500, graph_type="er", sem="lin", seed=9, er_p=2)
    n_edges = int(ds.adj_true.sum())
    # true adjacency is a permutation of an upper-triangular matrix, so it is a DAG
    assert n_edges >= 1


def test_sf_and_nonlin_paths_valid_and_deterministic():
    # scale-free and nonlinear SEM generators must produce finite, DAG-structured
    # data (the SF preferential-attachment graph is also upper-triangular in its
    # generation coordinate system).
    from data.generators import _sf_dag, make_benchmark_suite

    rng = np.random.default_rng(4)
    Bsf = _sf_dag(d=14, m=3, rng=rng)
    assert int(np.tril(Bsf, k=-1).sum()) == 0  # topological order preserved

    for sem in ("lin", "nonlin"):
        a = make_dataset(d=12, n=400, graph_type="sf", sem=sem, seed=7, sf_m=3)
        b = make_dataset(d=12, n=400, graph_type="sf", sem=sem, seed=7, sf_m=3)
        assert np.array_equal(a.X, b.X)
        assert np.isfinite(a.X).all()
        assert a.X.shape == (400, 12)
        assert int(a.adj_true.sum()) >= 1  # SF always has edges

    ds_nl = make_dataset(d=10, n=400, graph_type="er", sem="nonlin", seed=11, er_p=2)
    assert np.isfinite(ds_nl.X).all()
    assert ds_nl.X.std(axis=0).min() > 0.0

    suite = make_benchmark_suite(d=8, n=300, seeds=(1, 2, 3))
    assert len(suite) == 12  # 3 seeds x 2 graph types x 2 sems
    for ds in suite:
        assert ds.X.shape == (300, 8)
        assert ds.adj_true.shape == (8, 8)
