"""
Shared runtime helpers for multiprocessing-heavy 1D scripts.

These knobs keep Windows spawn-based worker startups from overcommitting BLAS
threads while SciPy and NumPy import in many child processes at once.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import sys

_THREAD_CAP_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def configure_numeric_runtime() -> None:
    """Cap per-process BLAS threads unless the caller already overrode them."""
    for name in _THREAD_CAP_VARS:
        os.environ.setdefault(name, "1")


def default_worker_count() -> int:
    """
    Return a conservative default worker count.

    Windows spawn imports are much heavier than fork-based startup, so we cap
    the default there to reduce SciPy DLL/page-file failures.
    """
    override = os.environ.get("EUROPA_MC_DEFAULT_WORKERS")
    if override is not None:
        try:
            return max(1, int(override))
        except ValueError:
            pass

    auto_workers = max(1, mp.cpu_count() - 1)
    return min(4, auto_workers) if sys.platform.startswith("win") else auto_workers


def resolve_worker_count(workers: int | None) -> int:
    """Validate an explicit worker request or fall back to the shared default."""
    if workers is None:
        return default_worker_count()
    if workers < 1:
        raise ValueError("workers must be >= 1")
    return workers
