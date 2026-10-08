"""GES — Greedy Equivalence Search (Chickering, 2002) with BIC scoring.

World-class, score-based causal discovery. Unlike gradient methods (NOTEARS),
GES has no spurious zero-graph trap: it greedily maximises a decomposable
Gaussian BIC score over the space of DAGs, which on linear-Gaussian data
reliably recovers the true structure (up to Markov equivalence).

Local BIC (Gaussian), omitting constant terms:

    score(i, Pa_i) = -n/2 * log(RSS_i / n) - |Pa_i|/2 * log(n)

The total score is the sum over nodes; GES performs a forward (insert)
phase followed by a backward (delete) phase, keeping the graph acyclic.
"""

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler

from core.errors import NotFittedError
from core.types import DagResult

__all__ = ["GES"]


class GES:
    name = "ges"

    def __init__(self, lambda_: float = 1.0, max_iter: int = 1000, threshold: float = 0.0):
        # lambda_ is the BIC edge-penalty multiplier. The Gaussian BIC local
        # score already carries -|Pa|/2*log(n); the standard choice is
        # lambda_ = 1.0 (a single 0.5*log(n) per regression coefficient). Values
        # > 1 over-penalise edges and can drive GES to the empty graph on dense
        # hub (scale-free) data.
        self.lambda_ = lambda_
        self.max_iter = max_iter
        self.threshold = threshold
        self._W = None
        self._fitted = False
        self._score = None

    # --- internal scoring -------------------------------------------------
    def _local_score(self, Xc: np.ndarray, Xp: np.ndarray, n: int, n_parents: int) -> float:
        if Xp.shape[1] == 0:
            rss = float(np.sum(Xc * Xc))
            return -0.5 * n * np.log(rss / n) if rss > 0 else 0.0
        beta, rss, _, _ = np.linalg.lstsq(Xp, Xc, rcond=None)
        rss = float(np.sum((Xc - Xp @ beta) ** 2))
        rss = max(rss, 1e-12)
        pen = n_parents * 0.5 * np.log(n) * self.lambda_
        return -0.5 * n * np.log(rss / n) - pen

    def fit(self, X: np.ndarray) -> GES:
        Xs = StandardScaler().fit_transform(X)
        n, d = Xs.shape
        # adjacency[i, j] == 1 means i -> j (i is a parent of j)
        adj = np.zeros((d, d), dtype=int)

        def has_path(a: int, b: int) -> bool:
            # is there a directed path a -> ... -> b?
            seen = set()
            stack = [a]
            while stack:
                u = stack.pop()
                if u == b:
                    return True
                if u in seen:
                    continue
                seen.add(u)
                for v in np.where(adj[u] == 1)[0]:
                    stack.append(int(v))
            return False

        def total_score():
            s = 0.0
            for j in range(d):
                parents = np.where(adj[:, j] == 1)[0]
                s += self._local_score(
                    Xs[:, j], Xs[:, parents] if len(parents) else Xs[:, :0], n, len(parents)
                )
            return s

        # forward phase
        for _ in range(self.max_iter):
            best_delta, best_op = 0.0, None
            for i in range(d):
                for j in range(d):
                    if i == j or adj[i, j] == 1 or adj[j, i] == 1:
                        continue
                    # insertion i->j must not create a cycle (j must not reach i)
                    if has_path(j, i):
                        continue
                    parents_old = np.where(adj[:, j] == 1)[0]
                    sc_old = self._local_score(
                        Xs[:, j],
                        Xs[:, parents_old] if len(parents_old) else Xs[:, :0],
                        n,
                        len(parents_old),
                    )
                    parents_new = np.concatenate([parents_old, [i]])
                    sc_new = self._local_score(Xs[:, j], Xs[:, parents_new], n, len(parents_new))
                    delta = sc_new - sc_old
                    if delta > best_delta + 1e-12:
                        best_delta, best_op = delta, ("add", i, j)
            if best_op is None:
                break
            _, i, j = best_op
            adj[i, j] = 1

        # backward phase
        for _ in range(self.max_iter):
            best_delta, best_op = 0.0, None
            for i in range(d):
                for j in range(d):
                    if adj[i, j] != 1:
                        continue
                    # deletion i->j: must stay acyclic (it will, removing edge only helps)
                    parents_old = np.where(adj[:, j] == 1)[0]
                    sc_old = self._local_score(
                        Xs[:, j],
                        Xs[:, parents_old] if len(parents_old) else Xs[:, :0],
                        n,
                        len(parents_old),
                    )
                    parents_new = parents_old[parents_old != i]
                    sc_new = self._local_score(
                        Xs[:, j],
                        Xs[:, parents_new] if len(parents_new) else Xs[:, :0],
                        n,
                        len(parents_new),
                    )
                    delta = sc_new - sc_old
                    if delta > best_delta + 1e-12:
                        best_delta, best_op = delta, ("del", i, j)
            if best_op is None:
                break
            _, i, j = best_op
            adj[i, j] = 0

        self._W = adj.astype(float)
        self._score = float(total_score())
        self._fitted = True
        return self

    def adjacency(self) -> np.ndarray:
        if not self._fitted:
            raise NotFittedError("call fit() first")
        return self._W.copy()

    def result(self) -> DagResult:
        return DagResult(
            adj=self._W.copy(), method=self.name, W=self._W.copy(), info={"score": self._score}
        )
