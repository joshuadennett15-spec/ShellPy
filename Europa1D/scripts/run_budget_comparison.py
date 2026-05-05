#!/usr/bin/env python3
"""
Run budget-constrained regional Monte Carlo comparison (Suite A).

Generates 7 result files:
    budget_global.npz                     — Global baseline (Howell 2021)
    budget_uniform_equator.npz            — Uniform, equator
    budget_uniform_pole.npz               — Uniform, pole
    budget_soderlund2014_equator.npz      — Soderlund 2014, equator
    budget_soderlund2014_pole.npz         — Soderlund 2014, pole
    budget_lemasquerier2023_equator.npz   — Lemasquerier 2023, equator
    budget_lemasquerier2023_pole.npz      — Lemasquerier 2023, pole

Usage:
    python run_budget_comparison.py              # all 7 ensembles
    python run_budget_comparison.py --skip-existing  # skip if .npz exists
    python run_budget_comparison.py -n 3000      # custom iteration count
"""
import sys
import os
import argparse
import multiprocessing as mp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from runtime_support import configure_numeric_runtime, resolve_worker_count

configure_numeric_runtime()

from Monte_Carlo import MonteCarloRunner, SolverConfig, HowellParameterSampler, save_results
from budget_samplers import (
    UniformEquatorSampler, UniformPoleSampler,
    Soderlund2014EquatorSampler, Soderlund2014PoleSampler,
    Lemasquerier2023EquatorSampler, Lemasquerier2023PoleSampler,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


# ── Ensemble definitions ────────────────────────────────────────────────────

ENSEMBLES = [
    ("budget_global",                   HowellParameterSampler,         False),
    ("budget_uniform_equator",          UniformEquatorSampler,          True),
    ("budget_uniform_pole",             UniformPoleSampler,             False),
    ("budget_soderlund2014_equator",    Soderlund2014EquatorSampler,    True),
    ("budget_soderlund2014_pole",       Soderlund2014PoleSampler,       False),
    ("budget_lemasquerier2023_equator", Lemasquerier2023EquatorSampler, True),
    ("budget_lemasquerier2023_pole",    Lemasquerier2023PoleSampler,    False),
]
# Third element: reject_subcritical flag
#   True  for equator  (subcritical Ra → unphysical thick conductive shell)
#   False for pole     (subcritical Ra → keep as purely conductive, which is valid)
#   False for global   (default Howell behaviour)


def run_ensemble(name, sampler_cls, reject_subcrit, n_iter, seed, n_workers,
                 skip_existing):
    outpath = os.path.join(RESULTS_DIR, f"{name}.npz")
    if skip_existing and os.path.exists(outpath):
        print(f"  [{name}] exists, skipping")
        return

    config = SolverConfig(reject_subcritical=reject_subcrit)
    runner = MonteCarloRunner(
        n_iterations=n_iter,
        seed=seed,
        verbose=True,
        n_workers=n_workers,
        config=config,
        sampler_class=sampler_cls,
    )

    print(f"\n{'='*60}")
    print(f"  {name}  (N={n_iter}, sampler={sampler_cls.__name__})")
    print(f"{'='*60}")

    results = runner.run()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    save_results(results, outpath)

    print(f"  CBE = {results.cbe_km:.1f} km, "
          f"median = {results.median_km:.1f} km, "
          f"valid = {results.n_valid}/{results.n_iterations}")


def main():
    parser = argparse.ArgumentParser(
        description="Run budget-constrained regional MC comparison")
    parser.add_argument("-n", type=int, default=5000,
                        help="Iterations per ensemble (default 5000)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=None,
                        help="Parallel workers (default: shared Windows-safe default)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip ensembles whose .npz already exists")
    args = parser.parse_args()

    n_workers = resolve_worker_count(args.workers)

    print(f"Budget-constrained regional comparison")
    print(f"  Iterations per ensemble: {args.n}")
    print(f"  Workers: {n_workers}")
    print(f"  Output: {RESULTS_DIR}/budget_*.npz")

    for name, sampler_cls, reject_subcrit in ENSEMBLES:
        run_ensemble(name, sampler_cls, reject_subcrit,
                     args.n, args.seed, n_workers, args.skip_existing)

    print(f"\nAll ensembles complete. Results in {RESULTS_DIR}/")


if __name__ == "__main__":
    mp.freeze_support()
    main()
