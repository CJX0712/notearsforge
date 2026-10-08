"""Weak baselines: empty graph and a random DAG."""

from __future__ import annotations

import numpy as np

from core.errors import NotFittedError
from core.types import DagResult

__all__ = ["EmptyGraph", "RandomDag"]


class EmptyGraph:
    name = "empty"
    _W = None
    _fitted = False

    def fit(self, X: np.ndarray) -> EmptyGraph:
        d = X.shape[1]
        self._W = np.zeros((d, d))
        self._fitted = True
        return self

    def adjacency(self) -> np.ndarray:
        if not self._fitted:
            raise NotFittedError("call fit() first")
        return self._W.copy()

    def result(self) -> DagResult:
        return DagResult(adj=self._W.copy(), method=self.name, W=self._W.copy())


class RandomDag:
    name = "random"
    _W = None
    _fitted = False

    def __init__(self, density: float = 0.2, seed: int = 0):
        self.density = density
        self.seed = seed

    def fit(self, X: np.ndarray) -> RandomDag:
        d = X.shape[1]
        rng = np.random.default_rng(self.seed + 7919)
        # random DAG: edge i->j with prob density for i<j (acyclic)
        B = (rng.random((d, d)) < self.density).astype(float)
        B = np.triu(B, k=1)
        self._W = B
        self._fitted = True
        return self

    def adjacency(self) -> np.ndarray:
        if not self._fitted:
            raise NotFittedError("call fit() first")
        return self._W.copy()

    def result(self) -> DagResult:
        return DagResult(adj=self._W.copy(), method=self.name, W=self._W.copy())
