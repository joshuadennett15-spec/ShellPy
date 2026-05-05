"""
Quick re-run: 1000 iterations with subcritical rejection filter.
Saves to results/monte_carlo_results.npz (overwrites).
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import multiprocessing as mp
from Monte_Carlo import MonteCarloRunner, save_results

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')

if __name__ == "__main__":
    mp.freeze_support()

    runner = MonteCarloRunner(
        n_iterations=1000,
        seed=42,
        verbose=True,
    )

    results = runner.run()
    save_results(results, os.path.join(RESULTS_DIR, "monte_carlo_results.npz"))
