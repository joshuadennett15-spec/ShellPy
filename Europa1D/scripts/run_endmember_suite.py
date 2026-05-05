"""
1D endmember proxy Monte Carlo suite.

Runs equatorial, mid-latitude (45°), and polar endmember proxies under
three ocean transport scenarios using audited 2026 priors with Andrade
rheology.

Produces 9 NPZ result files (3 scenarios x 3 endmembers).

q_tidal_multiplier values derived from 2D ocean_heat_flux() shape
functions at q_star=0.45:
  Uniform:            eq=1.00, mid=1.00, pole=1.00
  Soderlund 2014:     eq=1.15, mid=0.93, pole=0.70
  Lemasquerier 2023:  eq=0.85, mid=1.07, pole=1.30

reject_subcritical=False for all runs (no asymmetric branch handling).

Config must be set to Andrade in src/config.json.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import argparse
import multiprocessing as mp
from runtime_support import configure_numeric_runtime, resolve_worker_count

configure_numeric_runtime()

from Monte_Carlo import MonteCarloRunner, SolverConfig, save_results
from audited_endmember_sampler import (
    UniformEqSampler, UniformMidSampler, UniformPoleSampler,
    SoderlundEqSampler, SoderlundMidSampler, SoderlundPoleSampler,
    LemasquerierEqSampler, LemasquerierMidSampler, LemasquerierPoleSampler,
)
from constants import Rheology

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')

ENSEMBLES = [
    ("endmember_uniform_eq",          UniformEqSampler),
    ("endmember_uniform_mid",         UniformMidSampler),
    ("endmember_uniform_pole",        UniformPoleSampler),
    ("endmember_soderlund_eq",        SoderlundEqSampler),
    ("endmember_soderlund_mid",       SoderlundMidSampler),
    ("endmember_soderlund_pole",      SoderlundPoleSampler),
    ("endmember_lemasquerier_eq",     LemasquerierEqSampler),
    ("endmember_lemasquerier_mid",    LemasquerierMidSampler),
    ("endmember_lemasquerier_pole",   LemasquerierPoleSampler),
]


def run_ensemble(name, sampler_cls, n_iter, seed, n_workers, skip_existing):
    outpath = os.path.join(RESULTS_DIR, f"{name}_andrade.npz")
    if skip_existing and os.path.exists(outpath):
        print(f"  [{name}] exists, skipping")
        return

    config = SolverConfig(reject_subcritical=False)
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
        description="Run 1D endmember proxy MC suite")
    parser.add_argument("-n", type=int, default=5000,
                        help="Iterations per ensemble (default 5000)")
    parser.add_argument("--seed", type=int, default=10042)
    parser.add_argument("--workers", type=int, default=None,
                        help="Parallel workers (default: shared Windows-safe default)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip ensembles whose .npz already exists")
    args = parser.parse_args()

    n_workers = resolve_worker_count(args.workers)

    print(f"Rheology model: {Rheology.MODEL}")
    assert Rheology.MODEL == "Andrade", f"Expected Andrade, got {Rheology.MODEL}"

    print(f"1D Endmember Proxy Suite")
    print(f"  Iterations per ensemble: {args.n}")
    print(f"  Seed: {args.seed}")
    print(f"  Workers: {n_workers}")
    print(f"  Output: {RESULTS_DIR}/endmember_*.npz")

    for name, sampler_cls in ENSEMBLES:
        run_ensemble(name, sampler_cls, args.n, args.seed, n_workers,
                     args.skip_existing)

    print(f"\n{'='*60}")
    print("ENDMEMBER SUITE COMPLETE")
    for name, _ in ENSEMBLES:
        print(f"  {os.path.join(RESULTS_DIR, f'{name}_andrade.npz')}")
    print(f"{'='*60}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
