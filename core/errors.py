"""Error catalogue with stable codes (E100..E500).

Every raised error carries a machine-readable ``code`` so tests and the CLI
can branch on failure class rather than parsing messages.
"""

from __future__ import annotations


class ForgeError(Exception):
    code = "E000"
    description = "base error"


class DataError(ForgeError):
    code = "E100"
    description = "synthetic data / loader problem"


class NotFittedError(ForgeError):
    code = "E200"
    description = "estimator used before fit()"


class ConvergenceError(ForgeError):
    code = "E300"
    description = "optimizer failed to converge"


class ConfigError(ForgeError):
    code = "E400"
    description = "invalid configuration / schema violation"


class BackendUnavailable(ForgeError):
    code = "E500"
    description = "optional backend not installed"


ERROR_CATALOG = {
    e.code: e
    for e in (
        ForgeError,
        DataError,
        NotFittedError,
        ConvergenceError,
        ConfigError,
        BackendUnavailable,
    )
}


def error_class(code: str) -> type[ForgeError]:
    if code in ERROR_CATALOG:
        return ERROR_CATALOG[code]
    raise KeyError(f"unknown error code: {code}")
