"""Protocol contract for every structure-learning method.

Pipeline depends only on this interface; concrete algorithms live in
``causal.*`` and must honour the "score larger = more anomalous / fit first"
convention used by the rest of the system.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class StructureLearner(Protocol):
    name: str

    def fit(self, X: np.ndarray) -> StructureLearner:
        """Fit on observations ``X`` (n, d). Returns self for chaining."""
        ...

    def adjacency(self) -> np.ndarray:
        """Return binary estimated adjacency (d, d) with 1 = i->j edge."""
        ...
