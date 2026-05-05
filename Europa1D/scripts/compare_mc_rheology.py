import sys
import os
import multiprocessing as mp
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from runtime_support import configure_numeric_runtime

configure_numeric_runtime()

import matplotlib.pyplot as plt
import numpy as np

from Monte_Carlo import MonteCarloRunner, save_results
from constants import Rheology
from ConfigManager import ConfigManager

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')

def run_mc_for_model(model_name: str, n_iterations: int) -> dict:
    print(f"\n{'='*60}")
    print(f"Running Monte Carlo: {model_name} ({n_iterations} iterations)")
    print(f"{'='*60}")
    
    # Temporarily override config value
    ConfigManager()._config["rheology"] = {"model": model_name}
    Rheology.MODEL = model_name
    
    # We must instantiate a fresh runner so it pulls the current Rheology.MODEL downstream
    runner = MonteCarloRunner(
        n_iterations=n_iterations,
        seed=42, # Keep seed consistent for identical sampling
        verbose=True,
    )
    
    # Run
    results = runner.run()
    
    # Save
    out_file = os.path.join(RESULTS_DIR, f"mc_{model_name.lower()}_{n_iterations}.npz")
    save_results(results, out_file)
    
    # Return dict of relevant plotting data
    return {
        "model": model_name,
        "cbe_km": results.cbe_km,
        "sigma_1_low_km": results.sigma_1_low_km,
        "sigma_1_high_km": results.sigma_1_high_km,
        "pdf_x": results.histogram_bins[:-1],
        "pdf_y": results.histogram_counts,
        "thicknesses_km": results.thicknesses_km
    }

def plot_comparison(maxwell_data: dict, andrade_data: dict):
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Bins
    bins = np.linspace(0, 100, 50)
    
    # Maxwell Histogram
    ax.hist(maxwell_data["thicknesses_km"], bins=bins, density=True, 
            alpha=0.6, color='#1f77b4', label=f'Maxwell (CBE: {maxwell_data["cbe_km"]:.1f} km)')
            
    # Andrade Histogram
    ax.hist(andrade_data["thicknesses_km"], bins=bins, density=True, 
            alpha=0.6, color='#ff7f0e', label=f'Andrade (CBE: {andrade_data["cbe_km"]:.1f} km)')
            
    # Vertical CBE lines
    ax.axvline(maxwell_data["cbe_km"], color='#1f77b4', linestyle='dashed', linewidth=2)
    ax.axvline(andrade_data["cbe_km"], color='#ff7f0e', linestyle='dashed', linewidth=2)
    
    ax.set_xlabel('Ice Shell Thickness (km)', fontsize=12)
    ax.set_ylabel('Probability Density', fontsize=12)
    ax.set_title('Monte Carlo Comparison: Maxwell vs Andrade Rheology', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "mc_rheology_comparison.png")
    fig.savefig(out_path, dpi=300)

    print(f"\nSaved comparison plot to {out_path}")

if __name__ == "__main__":
    mp.freeze_support() # Required for Windows multiprocessing
    
    ITERATIONS = 1000
    
    # Run Maxwell
    data_maxwell = run_mc_for_model("Maxwell", ITERATIONS)
    
    # Run Andrade
    data_andrade = run_mc_for_model("Andrade", ITERATIONS)
    
    # Plot
    plot_comparison(data_maxwell, data_andrade)
    
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    print(f"Maxwell CBE: {data_maxwell['cbe_km']:.1f} km (1-sigma: {data_maxwell['sigma_1_low_km']:.1f} - {data_maxwell['sigma_1_high_km']:.1f})")
    print(f"Andrade CBE: {data_andrade['cbe_km']:.1f} km (1-sigma: {data_andrade['sigma_1_low_km']:.1f} - {data_andrade['sigma_1_high_km']:.1f})")
    print("="*60)
