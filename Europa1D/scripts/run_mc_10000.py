"""
Run 10000 Monte Carlo iterations with the transient solver.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import multiprocessing as mp
from Monte_Carlo import MonteCarloRunner, save_results

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')

if __name__ == "__main__":
    mp.freeze_support()
    
    print("=" * 60)
    print("Monte Carlo Simulation (10,000 iterations)")
    print("=" * 60)
    
    runner = MonteCarloRunner(
        n_iterations=10000,
        seed=42,
        verbose=True,
    )
    
    results = runner.run()
    save_results(results, os.path.join(RESULTS_DIR, "monte_carlo_results.npz"))
    
    print()
    print("=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(f"Target CBE (Howell 2021): ~24.3 km")
    print(f"Computed CBE: {results.cbe_km:.1f} km")
    print(f"1-sigma range: [{results.sigma_1_low_km:.1f}, {results.sigma_1_high_km:.1f}] km")
