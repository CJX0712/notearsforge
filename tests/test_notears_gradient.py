"""NOTEARS acyclicity-constraint gradient: verified vs finite differences.

The constraint h(W) = tr((I + W*W/d)^d) - d has a non-symmetric matrix
A = I + W*W/d, so its gradient is 2 * W * (A^{d-1})^T -- the transpose is
mandatory. A missing transpose is the single most common NOTEARS bug (it broke
the first solver we wrote). This test locks the correct analytic gradient to
~1e-9 against central finite differences.
"""

from __future__ import annotations

import numpy as np

from causal.notears_linear import _h_and_gh


def _finite_diff_grad(W, eps=1e-6):
    d = W.shape[0]
    G = np.zeros_like(W)
    for i in range(d):
        for j in range(d):
            e = np.zeros_like(W)
            e[i, j] = eps
            hp, _ = _h_and_gh(W + e)
            hm, _ = _h_and_gh(W - e)
            G[i, j] = (hp - hm) / (2 * eps)
    return G


def test_notears_gradient_matches_finite_differences():
    rng = np.random.default_rng(0)
    for d in (4, 6, 8):
        W = rng.normal(size=(d, d)) * 0.3
        h, grad = _h_and_gh(W)
        Gfd = _finite_diff_grad(W)
        rel = np.max(np.abs(grad - Gfd)) / max(1.0, np.max(np.abs(Gfd)))
        assert rel < 1e-6, f"gradient mismatch d={d}: max rel err {rel:.2e}"
        # h(W) = 0 iff W is a DAG; for a random dense W it is not zero
        assert h != 0.0


def test_notears_acyclicity_zero_for_dag():
    # strictly upper-triangular W is a DAG -> h(W) == 0
    d = 6
    W = np.triu(np.random.default_rng(1).normal(size=(d, d)), k=1) * 0.5
    h, _ = _h_and_gh(W)
    assert abs(h) < 1e-9, f"h(W) should be 0 for a DAG, got {h}"
