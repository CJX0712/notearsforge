"""Core seed control — single deterministic entry point for the whole system.

All randomness (``random``, ``numpy`` legacy, ``numpy.random.Generator``) is
seeded from one place so that ``run_demo`` is byte-for-byte reproducible.
"""

from __future__ import annotations

import random

import numpy as np

__all__ = ["get_seed", "set_all"]

_CURRENT_SEED: int | None = None


def set_all(seed: int = 42) -> np.random.Generator:
    """Seed every RNG source once and return a fresh ``Generator``.

    Using a single seed guarantees that two runs with the same seed produce
    identically distributed data and identical benchmark metrics (the
    determinism DoD gate).
    """
    global _CURRENT_SEED
    _CURRENT_SEED = int(seed)
    random.seed(_CURRENT_SEED)
    np.random.seed(_CURRENT_SEED)
    return np.random.default_rng(_CURRENT_SEED)


def get_seed() -> int | None:
    return _CURRENT_SEED
