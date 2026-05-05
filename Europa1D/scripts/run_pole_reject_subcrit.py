"""Re-run pole scenarios with reject_subcritical=True for comparison."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from runtime_support import configure_numeric_runtime, resolve_worker_count

configure_numeric_runtime()

import numpy as np
import multiprocessing as mp
import pandas as pd

from Monte_Carlo import MonteCarloRunner, SolverConfig, save_results
from regional_samplers_500 import Run1PoleSampler, Run2PoleSampler, Run3PoleSampler

def main(n_iterations=2500, n_workers=None, seed=1042):
    n_workers = resolve_worker_count(n_workers)
    base_kwargs = dict(
        nx=31, initial_thickness=20e3, dt=1e12, total_time=5e14,
        eq_threshold=1e-12, max_steps=1500, use_convection=True,
        rannacher_steps=4, use_warm_start=True,
    )
    # KEY CHANGE: reject_subcritical = True for pole
    pole_config = SolverConfig(**base_kwargs, reject_subcritical=True)

    scenarios = [
        ("Run 1 Pole (reject subcrit)", "run_1_polar_heating_pole_reject", Run1PoleSampler),
        ("Run 2 Pole (reject subcrit)", "run_2_eq_heating_pole_reject", Run2PoleSampler),
        ("Run 3 Pole (reject subcrit)", "run_3_uniform_pole_reject", Run3PoleSampler),
    ]

    results = []
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')

    for name, prefix, Sampler in scenarios:
        print(f"\n{'='*60}\n{name}\n{'='*60}")
        runner = MonteCarloRunner(
            n_iterations=n_iterations, seed=seed, verbose=False,
            n_workers=n_workers, config=pole_config, sampler_class=Sampler
        )
        res = runner.run()
        save_results(res, os.path.join(results_dir, f"{prefix}.npz"))

        results.append({
            "Scenario": name,
            "CBE (km)": res.cbe_km,
            "Mean (km)": res.mean_km,
            "Median (km)": res.median_km,
            "1-sigma Low (km)": res.sigma_1_low_km,
            "1-sigma High (km)": res.sigma_1_high_km,
            "Mean D_cond (km)": np.mean(res.D_cond_km),
            "Mean D_conv (km)": np.mean(res.D_conv_km),
        })

    df = pd.DataFrame(results)
    print("\n" + "="*80)
    print("POLE RESULTS (reject_subcritical=True, 2500 iterations)")
    print("="*80)
    print(df.to_markdown(index=False, floatfmt=".1f"))

if __name__ == "__main__":
    mp.freeze_support()
    main()
