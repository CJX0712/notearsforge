"""PC algorithm (Spirtes et al., 2000) — constraint-based causal discovery.

Fisher-z conditional independence tests on the correlation matrix, skeleton
recovery, v-structure orientation and a Meek-rule pass. Undirected edges in the
resulting CPDAG are represented bidirectionally (``adj[i,j]=adj[j,i]=1``) so
skeleton metrics see them as present.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy.stats import norm

from core.errors import NotFittedError
from core.types import DagResult

__all__ = ["PC"]


def _fisher_z(r: float, n: int, n_cond: int) -> float:
    r = max(min(r, 1.0 - 1e-12), -1.0 + 1e-12)
    z = 0.5 * np.log((1.0 + r) / (1.0 - r))
    return np.sqrt(max(n - 3 - n_cond, 1)) * z


class PC:
    name = "pc"

    def __init__(self, alpha: float = 0.05, max_cond: int = 8):
        self.alpha = alpha
        self.max_cond = max_cond
        self._W = None
        self._fitted = False

    def _corr(self, X: np.ndarray):
        d = X.shape[1]
        corr = np.zeros((d, d))
        for i in range(d):
            for j in range(d):
                if i == j:
                    corr[i, j] = 1.0
                else:
                    c = np.corrcoef(X[:, i], X[:, j])[0, 1]
                    corr[i, j] = 0.0 if np.isnan(c) else c
        return corr

    def fit(self, X: np.ndarray) -> PC:
        n, d = X.shape
        corr = self._corr(X)
        z_crit = norm.ppf(1.0 - self.alpha / 2.0)
        # adjacency (undirected skeleton, 1 = edge present either direction)
        skel = np.ones((d, d)) - np.eye(d)
        sepset = {(i, j): set() for i in range(d) for j in range(d) if i != j}

        # skeleton phase: grow the conditioning-set size and delete every edge
        # whose endpoints are conditionally independent. No early `break` -- at
        # ncond=0 (marginal independence) nothing is removed on correlated data,
        # yet the higher-order passes below are still required, so all ncond
        # levels must be attempted.
        for ncond in range(self.max_cond + 1):
            for i in range(d):
                for j in range(i + 1, d):
                    if skel[i, j] == 0:
                        continue
                    neighbors = [
                        k for k in range(d) if k not in (i, j) and skel[i, k] and skel[j, k]
                    ]
                    if len(neighbors) < ncond:
                        continue
                    for S in combinations(neighbors, ncond):
                        r = corr[i, j]
                        if S:
                            r = self._partial_corr(corr, i, j, list(S))
                        stat = abs(_fisher_z(r, n, len(S)))
                        if stat < z_crit:
                            skel[i, j] = skel[j, i] = 0
                            sepset[(i, j)] = set(S)
                            sepset[(j, i)] = set(S)
                            break

        # v-structure orientation: i - j - k, i _||_ k | sep, j not in sep
        directed = np.zeros((d, d), dtype=int)  # 1 = i->j
        for j in range(d):
            for i in range(d):
                if i == j or skel[i, j] == 0:
                    continue
                for k in range(d):
                    if k in (i, j) or skel[j, k] == 0 or skel[i, k] == 1:
                        continue
                    if j not in sepset[(i, k)]:
                        directed[i, j] = 1
                        directed[k, j] = 1

        # Meek-like orientation passes (R1, R2)
        for _ in range(d):
            for i in range(d):
                for j in range(d):
                    if directed[i, j] == 1 and directed[j, i] == 0:
                        # R1: i->j and j-?-k (undirected) and i-?-k absent -> j->k
                        for k in range(d):
                            if k in (i, j):
                                continue
                            if (
                                skel[j, k]
                                and not skel[i, k]
                                and directed[j, k] == 0
                                and directed[k, j] == 0
                            ):
                                directed[j, k] = 1
                        # R2: i->j and j->k and i-?-k undirected -> i->k
                        for k in range(d):
                            if k in (i, j):
                                continue
                            if (
                                directed[j, k] == 1
                                and skel[i, k]
                                and directed[i, k] == 0
                                and directed[k, i] == 0
                            ):
                                directed[i, k] = 1

        # build adjacency: directed edges single, undirected (skel but not directed) bidirectional
        adj = np.zeros((d, d), dtype=int)
        for i in range(d):
            for j in range(d):
                if directed[i, j] == 1:
                    adj[i, j] = 1
                elif skel[i, j] == 1 and directed[j, i] == 0:
                    adj[i, j] = 1
                    adj[j, i] = 1  # undirected -> bidirectional
        self._W = adj.astype(float)
        self._fitted = True
        return self

    def _partial_corr(self, corr, i, j, S):
        S = list(S)
        sub = [i, j, *S]
        C = corr[np.ix_(sub, sub)]
        try:
            inv = np.linalg.pinv(C)
        except np.linalg.LinAlgError:
            return corr[i, j]
        if inv[0, 0] <= 0 or inv[1, 1] <= 0:
            return 0.0
        r = -inv[0, 1] / np.sqrt(inv[0, 0] * inv[1, 1])
        return max(min(r, 1.0 - 1e-12), -1.0 + 1e-12)

    def adjacency(self) -> np.ndarray:
        if not self._fitted:
            raise NotFittedError("call fit() first")
        return self._W.copy()

    def result(self) -> DagResult:
        return DagResult(
            adj=self._W.copy(), method=self.name, W=self._W.copy(), info={"alpha": self.alpha}
        )
