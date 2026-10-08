"""Structure-recovery metrics.

All metrics operate on binary adjacency matrices (no self-loops). We report
SHD (per-pair structural Hamming distance), directed precision/recall/F1, and
oriented precision (of the skeleton edges recovered, how many have the right
direction). Higher is better for precision/recall/F1/oriented_precision.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "directed_prf1",
    "evaluate",
    "oriented_precision",
    "shd",
    "skeleton_prf1",
    "skeleton_shd",
]


def shd(A: np.ndarray, B: np.ndarray) -> int:
    """Structural Hamming distance (per unordered pair).

    Counts insertions, deletions and reversals of directed edges; each costs 1.
    """
    d = A.shape[0]
    s = 0
    for i in range(d):
        for j in range(i + 1, d):
            a = (int(A[i, j]), int(A[j, i]))
            b = (int(B[i, j]), int(B[j, i]))
            if a != b:
                s += 1
    return s


def directed_prf1(A: np.ndarray, B: np.ndarray) -> tuple[float, float, float]:
    """Directed-edge precision/recall/F1 (correct orientation required)."""
    tp = int(np.sum((B == 1) & (A == 1)))
    fp = int(np.sum((B == 1) & (A == 0)))
    fn = int(np.sum((A == 1) & (B == 0)))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def oriented_precision(A: np.ndarray, B: np.ndarray) -> float:
    """Of skeleton edges recovered, fraction with correct orientation."""
    d = A.shape[0]
    skel_tp = 0
    oriented = 0
    for i in range(d):
        for j in range(i + 1, d):
            a_edge = (A[i, j] == 1) or (A[j, i] == 1)
            b_edge = (B[i, j] == 1) or (B[j, i] == 1)
            if a_edge and b_edge:
                skel_tp += 1
                if (A[i, j] == 1 and B[i, j] == 1) or (A[j, i] == 1 and B[j, i] == 1):
                    oriented += 1
    return oriented / skel_tp if skel_tp else 0.0


def skeleton_prf1(A: np.ndarray, B: np.ndarray) -> tuple[float, float, float]:
    """Skeleton (undirected-edge) precision/recall/F1 — direction ignored.

    Essential for a fair comparison with PC, which only recovers the
    equivalence class (some edges undirected).
    """
    d = A.shape[0]
    tp = fp = fn = 0
    for i in range(d):
        for j in range(i + 1, d):
            a = int(A[i, j] == 1 or A[j, i] == 1)
            b = int(B[i, j] == 1 or B[j, i] == 1)
            tp += a == 1 and b == 1
            fp += a == 0 and b == 1
            fn += a == 1 and b == 0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def skeleton_shd(A: np.ndarray, B: np.ndarray) -> int:
    """Structural Hamming distance on the skeleton (undirected edge set)."""
    d = A.shape[0]
    s = 0
    for i in range(d):
        for j in range(i + 1, d):
            a = int(A[i, j] == 1 or A[j, i] == 1)
            b = int(B[i, j] == 1 or B[j, i] == 1)
            if a != b:
                s += 1
    return s


def evaluate(adj_true: np.ndarray, adj_est: np.ndarray) -> dict[str, float]:
    p, r, f1 = directed_prf1(adj_true, adj_est)
    sp, sr, sf1 = skeleton_prf1(adj_true, adj_est)
    return {
        "shd": float(shd(adj_true, adj_est)),
        "precision": p,
        "recall": r,
        "f1": f1,
        "oriented_precision": oriented_precision(adj_true, adj_est),
        "skel_shd": float(skeleton_shd(adj_true, adj_est)),
        "skel_precision": sp,
        "skel_recall": sr,
        "skel_f1": sf1,
    }
