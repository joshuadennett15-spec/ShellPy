import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from runtime_support import configure_numeric_runtime, resolve_worker_count

configure_numeric_runtime()

import numpy as np
import multiprocessing as mp
import pandas as pd

from Monte_Carlo import MonteCarloRunner, SolverConfig, save_results
from regional_samplers_500 import (
    Run1EquatorSampler, Run1PoleSampler,
    Run2EquatorSampler, Run2PoleSampler,
    Run3EquatorSampler, Run3PoleSampler
)

def run_scenarios(n_iterations=2500, n_workers=None, seed=42):
    n_workers = resolve_worker_count(n_workers)
    base_kwargs = dict(
        nx=31,
        initial_thickness=20e3,
        dt=1e12,
        total_time=5e14,
        eq_threshold=1e-12,
        max_steps=1500,
        use_convection=True,
        rannacher_steps=4,
        use_warm_start=True,
    )
    
    # Equator usually gets subcritical rejection to allow thick shells to form properly
    equator_config = SolverConfig(**base_kwargs, reject_subcritical=True)
    pole_config = SolverConfig(**base_kwargs, reject_subcritical=False)
    
    scenarios = [
        ("Run 1 (Polar Heating)", Run1EquatorSampler, Run1PoleSampler),
        ("Run 2 (Eq Heating)", Run2EquatorSampler, Run2PoleSampler),
        ("Run 3 (Uniform)", Run3EquatorSampler, Run3PoleSampler)
    ]
    
    results = []
    
    for name, EqSampler, PoleSampler in scenarios:
        print(f"\n{'='*60}\nRunning {name}\n{'='*60}")
        
        # Equator
        print(f"--> Equator ({name})")
        eq_runner = MonteCarloRunner(
            n_iterations=n_iterations, seed=seed, verbose=False,
            n_workers=n_workers, config=equator_config, sampler_class=EqSampler
        )
        eq_res = eq_runner.run()
        
        # Pole
        print(f"--> Pole ({name})")
        pole_runner = MonteCarloRunner(
            n_iterations=n_iterations, seed=seed+1000, verbose=False,
            n_workers=n_workers, config=pole_config, sampler_class=PoleSampler
        )
        pole_res = pole_runner.run()
        
        # Save results arrays for plotting and analysis
        safe_name = name.replace(' ', '_').replace('(', '').replace(')', '').replace(':', '').lower()
        save_results(eq_res, os.path.join(os.path.dirname(__file__), '..', 'results', f"{safe_name}_equator.npz"))
        save_results(pole_res, os.path.join(os.path.dirname(__file__), '..', 'results', f"{safe_name}_pole.npz"))
        
        results.append({
            "Scenario": name,
            "Region": "Equator",
            "CBE (km)": eq_res.cbe_km,
            "Mean (km)": eq_res.mean_km,
            "Median (km)": eq_res.median_km,
            "1-sigma Low (km)": eq_res.sigma_1_low_km,
            "1-sigma High (km)": eq_res.sigma_1_high_km,
            "Mean D_cond (km)": np.mean(eq_res.D_cond_km)
        })
        
        results.append({
            "Scenario": name,
            "Region": "Pole",
            "CBE (km)": pole_res.cbe_km,
            "Mean (km)": pole_res.mean_km,
            "Median (km)": pole_res.median_km,
            "1-sigma Low (km)": pole_res.sigma_1_low_km,
            "1-sigma High (km)": pole_res.sigma_1_high_km,
            "Mean D_cond (km)": np.mean(pole_res.D_cond_km)
        })
        
    df = pd.DataFrame(results)
    
    # Format as markdown table
    md_table = df.to_markdown(index=False, floatfmt=".1f")
    
    print("\n\n" + "="*80)
    print("RESULTS SUMMARY (500 iterations each)")
    print("="*80)
    print(md_table)
    
    with open(os.path.join(os.path.dirname(__file__), '..', 'results', '500gw_scenarios_table.md'), 'w') as f:
        f.write("# 500 GW Budget Scenarios Results\n\n")
        f.write(md_table)
        f.write("\n")
        
if __name__ == "__main__":
    mp.freeze_support()
    run_scenarios()
