"""Synthetic causal datasets with *known* ground-truth DAGs.

Two graph families (Erdos-Renyi ``er``, scale-free ``sf``) and two SEM
families (linear-Gaussian ``lin`` and nonlinear ``nonlin``). Every sample
carries its true binary adjacency so structure recovery can be scored exactly.

Determinism: identical ``seed`` -> identical dataset (the generators consume a
``numpy.random.Generator`` produced by ``core.seed.set_all``).
"""

from __future__ import annotations

import numpy as np

from core.types import Dataset

__all__ = ["make_benchmark_suite", "make_dataset"]


def _er_dag(d: int, p: float, rng: np.random.Generator) -> np.ndarray:
    """Upper-triangular acyclic adjacency: parents have LOWER index than child.

    The SEM (data/generators._linear_sem) assigns node values in index order,
    so a parent must already be evaluated when its child is generated. Using
    ``triu`` (nonzero only where row < col) guarantees parent index < child
    index. Using ``tril`` here would silently produce noise-only data.
    """
    prob = min(1.0, max(1e-6, p / (d - 1)))
    B = rng.choice([0, 1], size=(d, d), p=[1 - prob, prob]).astype(float)
    B = np.triu(B, k=1)
    return B


def _sf_dag(d: int, m: int, rng: np.random.Generator) -> np.ndarray:
    """Preferential-attachment DAG oriented along topological order.

    Edge convention is ``B[parent, child] = 1`` with **parent index < child
    index**, identical to ``_er_dag`` -- so the SEM (which evaluates nodes in
    index order) always has a parent's value available when it generates the
    child. (Writing ``B[child, parent]`` here would silently produce noise-only
    data, the same failure mode as using ``tril`` in ``_er_dag``.)
    """
    B = np.zeros((d, d))
    m = max(m, 1)
    # seed clique: node k gets every earlier node 0..k-1 as a parent
    for k in range(1, m):
        for j in range(k):
            B[j, k] = 1.0
    # preferential attachment (Barabasi-Albert): new node v attaches to m earlier
    # nodes with probability proportional to their current in-degree (+1).
    targets = list(range(m))  # existing nodes; every one has index < v
    for v in range(m, d):
        degs = np.array([1.0 + B[:, t].sum() for t in targets])
        degs /= degs.sum()
        parents = rng.choice(targets, size=min(m, len(targets)), replace=False, p=degs)
        for t in parents:
            B[t, v] = 1.0  # parent t -> child v (t < v -> topological order)
        targets.append(v)
    return B


def _linear_sem(
    B: np.ndarray, n: int, rng: np.random.Generator, noise: float = 1.0, w_range: tuple = (0.5, 1.5)
) -> np.ndarray:
    d = B.shape[0]
    W = B * rng.uniform(*w_range, size=(d, d)) * rng.choice([-1.0, 1.0], size=(d, d))
    X = np.zeros((n, d))
    for j in range(d):  # topological order
        parents = np.where(B[:, j] == 1)[0]
        if len(parents):
            X[:, j] = X[:, parents] @ W[parents, j]
        X[:, j] += rng.normal(0.0, noise, size=n)
    return X


def _nonlinear_sem(
    B: np.ndarray, n: int, rng: np.random.Generator, noise: float = 0.6, w_range: tuple = (0.6, 1.2)
) -> np.ndarray:
    d = B.shape[0]
    W = B * rng.uniform(*w_range, size=(d, d))
    X = np.zeros((n, d))
    for j in range(d):  # topological order
        parents = np.where(B[:, j] == 1)[0]
        z = X[:, parents] @ W[parents, j] if len(parents) else np.zeros(n)
        # smooth, invertible-ish nonlinear mixing of the parental sum
        g = np.sin(z) + np.cos(0.7 * z + 0.3) + 0.5 * np.tanh(1.2 * z)
        X[:, j] = g + rng.normal(0.0, noise, size=n)
    return X


def make_dataset(
    *,
    d: int,
    n: int,
    graph_type: str,
    sem: str,
    seed: int,
    er_p: int = 2,
    sf_m: int = 3,
    noise: float | None = None,
) -> Dataset:
    """Generate one synthetic dataset with known ground-truth adjacency."""
    rng = np.random.default_rng(seed + d * 1000 + (0 if graph_type == "er" else 7))
    if graph_type == "er":
        B = _er_dag(d, float(er_p), rng)
    elif graph_type == "sf":
        B = _sf_dag(d, sf_m, rng)
    else:
        raise ValueError(f"unknown graph_type: {graph_type}")
    if sem == "lin":
        X = _linear_sem(B, n, rng, noise=noise if noise else 1.0)
    elif sem == "nonlin":
        X = _nonlinear_sem(B, n, rng, noise=noise if noise else 0.6)
    else:
        raise ValueError(f"unknown sem: {sem}")
    # randomise node ordering so the DAG is not trivially lower-triangular
    perm = rng.permutation(d)
    X = X[:, perm]
    adj_true = B[np.ix_(perm, perm)]
    name = f"{graph_type}_{sem}_d{d}_n{n}_s{seed}"
    return Dataset(
        X=X, adj_true=adj_true, name=name, sem=sem, graph_type=graph_type, n=n, d=d, seed=seed
    )


def make_benchmark_suite(
    d: int = 20, n: int = 1000, seeds: tuple = (7, 11, 23), er_p: int = 2, sf_m: int = 3
) -> list[Dataset]:
    """Standard suite: ER + SF crossed with linear + nonlinear SEMs."""
    suite: list[Dataset] = []
    for seed in seeds:
        for graph_type in ("er", "sf"):
            for sem in ("lin", "nonlin"):
                suite.append(
                    make_dataset(
                        d=d, n=n, graph_type=graph_type, sem=sem, seed=seed, er_p=er_p, sf_m=sf_m
                    )
                )
    return suite
