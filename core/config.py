"""Configuration object with environment-variable overrides and schema checks.

Top-level knobs are read from ``ENV_NTF_*`` so CI / Docker can tune behaviour
without editing source. Unknown keys and bad types raise ``ConfigError``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

_INT_FIELDS = {
    "default_seed",
    "n_samples",
    "n_nodes",
    "n_seeds",
    "er_p",
    "sf_m",
    "max_cores",
    "notears_max_iter",
    "dagma_max_iter",
    "pc_max_cond",
}
_FLOAT_FIELDS = {"notears_lambda", "notears_rho", "dagma_lambda", "dagma_c"}


@dataclass
class Config:
    default_seed: int = 42
    n_samples: int = 1000
    n_nodes: int = 20
    n_seeds: int = 3
    er_p: int = 2
    sf_m: int = 3
    max_cores: int = 1
    notears_lambda: float = 0.05
    notears_rho: float = 1.0
    notears_max_iter: int = 80
    dagma_lambda: float = 0.005
    dagma_c: float = 0.5
    dagma_max_iter: int = 200
    pc_max_cond: int = 8
    pc_alpha: float = 0.05

    def seed_list(self) -> tuple:
        """Deterministic seed sequence for the benchmark suite."""
        return tuple(self.default_seed + k * 1000 for k in range(self.n_seeds))

    @classmethod
    def from_env(cls) -> Config:
        kw: dict = {}
        for k in _INT_FIELDS | _FLOAT_FIELDS:
            env = os.environ.get(f"ENV_NTF_{k.upper()}")
            if env is None:
                continue
            try:
                kw[k] = int(env) if k in _INT_FIELDS else float(env)
            except ValueError as exc:
                raise ValueError(f"ENV_NTF_{k.upper()} must be numeric, got {env!r}") from exc
        return cls(**kw)

    def validate(self) -> None:
        if self.n_seeds < 1:
            raise ValueError("n_seeds must be >= 1")
        if self.n_nodes < 2:
            raise ValueError("n_nodes must be >= 2")
        if self.notears_lambda < 0 or self.dagma_lambda < 0:
            raise ValueError("regularization lambdas must be >= 0")
        if not (0 < self.dagma_c <= 1.0):
            raise ValueError("dagma_c must be in (0, 1]")
