"""Tests for the structure-recovery metrics and the two flagship algorithms on
a tiny, hand-checked linear-Gaussian DAG.
"""

from __future__ import annotations

import numpy as np

from causal.greedy import GES
from causal.pc import PC
from eval.metrics import (
    directed_prf1,
    evaluate,
    shd,
    skeleton_prf1,
)


def test_metrics_known_example():
    # chain 0 -> 1 -> 2
    A = np.zeros((3, 3))
    A[0, 1] = A[1, 2] = 1
    # estimated: same edges plus a false edge 0->2
    B = np.zeros((3, 3))
    B[0, 1] = B[1, 2] = B[0, 2] = 1

    assert shd(A, B) == 1  # one reversal/insertion (0->2)
    p, r, f1 = directed_prf1(A, B)
    assert p == 2 / 3
    assert r == 1.0
    assert abs(f1 - 0.8) < 1e-9

    sp, sr, sf1 = skeleton_prf1(A, B)
    # B has 3 skeleton edges ({0-1},{1-2},{0-2}); 2 are true -> precision 2/3
    assert abs(sp - 2 / 3) < 1e-9
    assert sr == 1.0  # all true skeleton edges present
    assert abs(sf1 - 0.8) < 1e-9


def test_ges_recovers_small_chain():
    # deterministic 4-node chain in topological order, strongly identifiable
    rng = np.random.default_rng(5)
    d, n = 6, 800
    B = np.zeros((d, d))
    for j in range(1, d):
        B[j - 1, j] = 1.0  # j-1 -> j
    X = np.zeros((n, d))
    W = np.zeros((d, d))
    for j in range(d):
        parents = np.where(B[:, j] == 1)[0]
        if len(parents):
            w = rng.uniform(0.8, 1.2, size=len(parents))
            W[parents, j] = w * rng.choice([-1.0, 1.0], size=len(parents))
            X[:, j] = X[:, parents] @ W[parents, j]
        X[:, j] += rng.normal(0, 0.3, size=n)
    adj = GES().fit(X).adjacency()
    res = evaluate(B, adj)
    assert res["skel_f1"] >= 0.8


def test_pc_recovers_small_chain():
    rng = np.random.default_rng(6)
    d, n = 6, 800
    B = np.zeros((d, d))
    for j in range(1, d):
        B[j - 1, j] = 1.0
    X = np.zeros((n, d))
    W = np.zeros((d, d))
    for j in range(d):
        parents = np.where(B[:, j] == 1)[0]
        if len(parents):
            w = rng.uniform(0.8, 1.2, size=len(parents))
            W[parents, j] = w * rng.choice([-1.0, 1.0], size=len(parents))
            X[:, j] = X[:, parents] @ W[parents, j]
        X[:, j] += rng.normal(0, 0.3, size=n)
    adj = PC().fit(X).adjacency()
    res = evaluate(B, adj)
    assert res["skel_f1"] >= 0.6
