"""
Run 15,000 Andrade MC with audited shell-level priors (Option A).
Config must already be set to Andrade in src/config.json.
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import multiprocessing as mp
from runtime_support import configure_numeric_runtime, default_worker_count, resolve_worker_count

configure_numeric_runtime()

from Monte_Carlo import MonteCarloRunner, SolverConfig, save_results
from audited_sampler import AuditedShellSampler
from constants import Rheology

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')


def _default_workers() -> int:
    """Use the shared Windows-safe worker default for 1D Monte Carlo runs."""
    return default_worker_count()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--iterations",
        type=int,
        default=15000,
        help="Number of Monte Carlo iterations (default: 15000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=10042,
        help="Random seed (default: 10042).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=_default_workers(),
        help="Parallel worker processes. Windows default is capped conservatively.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    mp.freeze_support()
    args = _parse_args()
    workers = resolve_worker_count(args.workers)

    print(f"Rheology model: {Rheology.MODEL}")
    assert Rheology.MODEL == "Andrade", f"Expected Andrade, got {Rheology.MODEL}"
    print(f"Workers: {workers}")

    config = SolverConfig(reject_subcritical=False)

    runner = MonteCarloRunner(
        n_iterations=args.iterations,
        seed=args.seed,
        verbose=True,
        n_workers=workers,
        config=config,
        sampler_class=AuditedShellSampler,
    )
    results = runner.run()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    save_results(results, os.path.join(RESULTS_DIR, "mc_15000_optionA_v2_andrade.npz"))

    print(f"\nCBE: {results.cbe_km:.1f} km")
    print(f"Median: {results.median_km:.1f} km")
    print(f"1-sigma: [{results.sigma_1_low_km:.1f}, {results.sigma_1_high_km:.1f}] km")
    print(f"Valid: {results.n_valid}/{results.n_iterations}")
