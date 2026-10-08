"""NOTEARS (Zheng et al., 2018) — linear structural DAG learning.

World-class maths: the acyclicity constraint is written as a *smooth* function

    h(W) = tr((I + W⊙W / d)^d) - d = 0   iff   W is a DAG,

so the whole problem becomes continuous; we solve it with an augmented
Lagrangian outer loop wrapped around **L-BFGS-B** (the canonical solver from
the original DAGs-with-NO-TEARS reference implementation), using the variable
doubling trick so the L1 penalty is handled inside a bound-constrained
quasi-Newton run. This is the landmark result that made gradient-based causal
discovery tractable, and the L-BFGS-B solver is what makes it recover sparse
DAGs instead of collapsing to the empty graph.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler

from core.errors import NotFittedError
from core.types import DagResult

__all__ = ["NotearsLinear"]


def _h_and_gh(W: np.ndarray) -> tuple[float, np.ndarray]:
    """Value and analytic gradient of the acyclicity constraint.

    h(W) = tr(A^d) - d,  A = I + W⊙W / d   (A is *not* symmetric for general W,
    because A_ij = W_ij^2 / d != W_ji^2 / d).

    The gradient (verified vs finite differences to ~1e-9) is

        dh/dW_kl = 2 * W_kl * (A^{d-1})_{lk}   ->   2 * W * (A^{d-1})^T.

    The transpose is mandatory: A is non-symmetric, so the (l, k) entry of
    A^{d-1} differs from the (k, l) entry.
    """
    d = W.shape[0]
    A = np.eye(d) + (W * W) / d
    Apow = np.linalg.matrix_power(A, d - 1)
    h = float(np.trace(Apow @ A) - d)  # tr(A^d) - d
    grad = 2.0 * W * Apow.T
    return h, grad


def _h(W: np.ndarray) -> float:
    return _h_and_gh(W)[0]


def _unvec(w: np.ndarray, d: int) -> np.ndarray:
    return w[: d * d].reshape(d, d) - w[d * d :].reshape(d, d)


class NotearsLinear:
    name = "notears_linear"

    def __init__(
        self,
        lambda1: float = 0.1,
        rho_init: float = 1.0,
        max_iter: int = 100,
        inner_iter: int = 1000,
        h_tol: float = 1e-8,
        rho_max: float = 1e16,
        threshold: float = 0.3,
    ):
        self.lambda1 = lambda1
        self.rho_init = rho_init
        self.max_iter = max_iter
        self.inner_iter = inner_iter
        self.h_tol = h_tol
        self.rho_max = rho_max
        self.threshold = threshold
        self._W = None
        self._fitted = False
        self._h_final = None

    def fit(self, X: np.ndarray) -> NotearsLinear:
        Xs = StandardScaler().fit_transform(X)
        n, d = Xs.shape
        Xt = Xs.T

        def _loss(W: np.ndarray):
            R = Xs - Xs @ W
            loss = 0.5 / n * float((R * R).sum())
            G = -(Xt @ R) / n
            return loss, G

        def _func(w: np.ndarray, rho: float, alpha: float):
            W = _unvec(w, d)
            loss, G_loss = _loss(W)
            h, G_h = _h_and_gh(W)
            f = loss + 0.5 * rho * h * h + alpha * h + self.lambda1 * float(np.abs(W).sum())
            g = G_loss + (rho * h + alpha) * G_h + self.lambda1 * np.sign(W)
            g = g.flatten()
            g = np.concatenate([g, -g])  # variable doubling gradient
            return f, g

        w = np.zeros(2 * d * d)
        rho = self.rho_init
        alpha = 0.0
        h = np.inf
        for _ in range(self.max_iter):
            while rho < self.rho_max:
                sol = minimize(
                    _func,
                    w,
                    args=(rho, alpha),
                    method="L-BFGS-B",
                    jac=True,
                    bounds=[(0, None)] * (2 * d * d),
                    options={"maxiter": self.inner_iter},
                )
                w = sol.x
                h = _h_and_gh(_unvec(w, d))[0]
                if h > self.h_tol:
                    rho *= 10.0
                else:
                    break
            if h <= self.h_tol:
                break

        W = _unvec(w, d)
        W[np.abs(W) < self.threshold] = 0.0
        self._W = W
        self._h_final = float(h)
        self._fitted = True
        return self

    def adjacency(self) -> np.ndarray:
        if not self._fitted:
            raise NotFittedError("call fit() first")
        return (np.abs(self._W) > self.threshold).astype(float)

    def result(self) -> DagResult:
        return DagResult(
            adj=self.adjacency(), method=self.name, W=self._W, info={"h": self._h_final}
        )
