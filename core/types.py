"""Shared dataclasses and small numeric helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Dataset:
    """A synthetic dataset with a known ground-truth DAG adjacency."""

    X: np.ndarray
    adj_true: np.ndarray
    name: str
    sem: str
    graph_type: str
    n: int
    d: int
    seed: int

    def __post_init__(self) -> None:
        if self.X.ndim != 2:
            raise ValueError("X must be 2-D (n, d)")
        if self.adj_true.shape != (self.d, self.d):
            raise ValueError("adj_true must be (d, d)")


@dataclass
class DagResult:
    """Result of a structure-learning run."""

    adj: np.ndarray
    method: str
    W: np.ndarray | None = None
    info: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.adj.ndim != 2 or self.adj.shape[0] != self.adj.shape[1]:
            raise ValueError("adj must be a square matrix")


@dataclass
class BenchmarkRecord:
    method: str
    dataset: str
    sem: str
    graph_type: str
    seed: int
    shd: float
    precision: float
    recall: float
    f1: float
    oriented_precision: float
    elapsed_sec: float
    n_edges: int
    skipped: bool = False
    note: str = ""


def symmetrize(W: np.ndarray) -> np.ndarray:
    return 0.5 * (W + W.T)
