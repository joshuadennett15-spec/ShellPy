"""
Polar-proxy Monte Carlo run with audited priors.

Mirrors run_equatorial_suite.py but uses polar surface conditions
(T_surf ~ 50 K, higher epsilon_0) with no ocean heat transport scaling.
Produces one NPZ for downstream comparison against the equatorial suite.

Shared seed: matches the equatorial suite seed so that non-overridden
parameter draws (d_grain, Q_v, Q_b, etc.) come from the same RNG
sequence, reducing sampler-side noise in eq-vs-pole comparisons.

Config must be set to Andrade in src/config.json.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import multiprocessing as mp
from runtime_support import configure_numeric_runtime, default_worker_count

configure_numeric_runtime()

from Monte_Carlo import MonteCarloRunner, SolverConfig, save_results
from audited_polar_sampler import AuditedPolarSampler
from constants import Rheology

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')

SEED = 10042
N_ITERATIONS = 15000


# Module-level sampler class (picklable on Windows spawn-based mp)
class PolarSampler(AuditedPolarSampler):
    """Polar proxy: T_surf ~ N(50,5), epsilon_0 ~ 1.2e-5."""
    pass


def main():
    n_workers = default_worker_count()

    print(f"Rheology model: {Rheology.MODEL}")
    assert Rheology.MODEL == "Andrade", f"Expected Andrade, got {Rheology.MODEL}"

    print(f"\n{'=' * 60}")
    print(f"POLAR PROXY: T_surf ~ N(50,5) K, epsilon_0 ~ 1.2e-5")
    print(f"  N = {N_ITERATIONS:,}, seed = {SEED} (paired with equatorial suite)")
    print(f"  workers = {n_workers}")
    print(f"{'=' * 60}")

    config = SolverConfig(reject_subcritical=False)

    runner = MonteCarloRunner(
        n_iterations=N_ITERATIONS,
        seed=SEED,
        verbose=True,
        n_workers=n_workers,
        config=config,
        sampler_class=PolarSampler,
    )
    results = runner.run()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_path = os.path.join(RESULTS_DIR, "polar_andrade.npz")
    save_results(results, output_path)

    print(f"\n--- POLAR PROXY RESULTS ---")
    print(f"  CBE:     {results.cbe_km:.1f} km")
    print(f"  Median:  {results.median_km:.1f} km")
    print(f"  1-sigma: [{results.sigma_1_low_km:.1f}, {results.sigma_1_high_km:.1f}] km")
    print(f"  Valid:   {results.n_valid}/{results.n_iterations}")

    if results.subpopulations:
        for sub in results.subpopulations:
            print(f"  {sub.label}: {sub.fraction*100:.0f}% (n={sub.n_samples})")

    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
