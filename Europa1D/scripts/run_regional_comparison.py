"""
Regional Comparison: Equator vs Pole Ice Shell Structure

Runs two Monte Carlo simulations with different regional parameters:
1. EQUATOR: Warm surface, low tidal strain, low basal flux
2. POLE: Cold surface, high tidal strain, high basal flux

Outputs comparison statistics and saves results for further analysis.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from runtime_support import configure_numeric_runtime, default_worker_count

configure_numeric_runtime()

import numpy as np
import multiprocessing as mp
from typing import Dict, Any

from Monte_Carlo import MonteCarloRunner, MonteCarloResults, SolverConfig, save_results
from regional_samplers import EquatorParameterSampler, PoleParameterSampler

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')


def print_comparison(equator_results: MonteCarloResults, 
                     pole_results: MonteCarloResults) -> None:
    """Print side-by-side comparison of equator vs pole results."""
    
    print("\n" + "=" * 70)
    print("           EUROPA ICE SHELL: EQUATOR vs POLE COMPARISON")
    print("=" * 70)
    
    # Simulation quality
    print("\n--- SIMULATION QUALITY ---")
    print(f"{'Metric':<25} {'Equator':>20} {'Pole':>20}")
    print("-" * 65)
    print(f"{'Valid samples':<25} {equator_results.n_valid:>15} / {equator_results.n_iterations} "
          f"{pole_results.n_valid:>15} / {pole_results.n_iterations}")
    print(f"{'Runtime (s)':<25} {equator_results.runtime_seconds:>20.1f} {pole_results.runtime_seconds:>20.1f}")
    
    # Ice shell thickness
    print("\n--- ICE SHELL THICKNESS ---")
    print(f"{'Statistic':<25} {'Equator (km)':>20} {'Pole (km)':>20}")
    print("-" * 65)
    print(f"{'CBE (mode)':<25} {equator_results.cbe_km:>20.1f} {pole_results.cbe_km:>20.1f}")
    print(f"{'Median':<25} {equator_results.median_km:>20.1f} {pole_results.median_km:>20.1f}")
    print(f"{'Mean':<25} {equator_results.mean_km:>20.1f} {pole_results.mean_km:>20.1f}")
    print(f"{'1-sigma low':<25} {equator_results.sigma_1_low_km:>20.1f} {pole_results.sigma_1_low_km:>20.1f}")
    print(f"{'1-sigma high':<25} {equator_results.sigma_1_high_km:>20.1f} {pole_results.sigma_1_high_km:>20.1f}")
    
    # Convection structure
    if equator_results.lid_fractions is not None and pole_results.lid_fractions is not None:
        print("\n--- CONVECTION STRUCTURE ---")
        print(f"{'Parameter':<25} {'Equator':>20} {'Pole':>20}")
        print("-" * 65)
        
        eq_lid = np.mean(equator_results.lid_fractions)
        pole_lid = np.mean(pole_results.lid_fractions)
        print(f"{'Mean lid fraction':<25} {eq_lid:>19.1%} {pole_lid:>19.1%}")
        
        eq_Dcond = np.mean(equator_results.D_cond_km)
        pole_Dcond = np.mean(pole_results.D_cond_km)
        print(f"{'Mean D_cond (km)':<25} {eq_Dcond:>20.1f} {pole_Dcond:>20.1f}")
        
        eq_Dconv = np.mean(equator_results.D_conv_km)
        pole_Dconv = np.mean(pole_results.D_conv_km)
        print(f"{'Mean D_conv (km)':<25} {eq_Dconv:>20.1f} {pole_Dconv:>20.1f}")
        
        eq_Ra = np.mean(equator_results.Ra_values)
        pole_Ra = np.mean(pole_results.Ra_values)
        print(f"{'Mean Ra':<25} {eq_Ra:>20.2e} {pole_Ra:>20.2e}")
        
        eq_Nu = np.mean(equator_results.Nu_values)
        pole_Nu = np.mean(pole_results.Nu_values)
        print(f"{'Mean Nu':<25} {eq_Nu:>20.1f} {pole_Nu:>20.1f}")
    
    # Input parameter summary
    print("\n--- INPUT PARAMETERS (means) ---")
    print(f"{'Parameter':<25} {'Equator':>20} {'Pole':>20}")
    print("-" * 65)
    
    eq_Tsurf = np.mean(equator_results.sampled_params['T_surf'])
    pole_Tsurf = np.mean(pole_results.sampled_params['T_surf'])
    print(f"{'T_surf (K)':<25} {eq_Tsurf:>20.1f} {pole_Tsurf:>20.1f}")
    
    eq_eps = np.mean(equator_results.sampled_params['epsilon_0'])
    pole_eps = np.mean(pole_results.sampled_params['epsilon_0'])
    print(f"{'epsilon_0':<25} {eq_eps:>20.2e} {pole_eps:>20.2e}")
    
    eq_Ptidal = np.mean(equator_results.sampled_params['P_tidal'])
    pole_Ptidal = np.mean(pole_results.sampled_params['P_tidal'])
    print(f"{'P_tidal (GW)':<25} {eq_Ptidal/1e9:>20.1f} {pole_Ptidal/1e9:>20.1f}")
    
    # Estimated basal flux
    A_surface = 4 * np.pi * (1.561e6) ** 2  # Europa surface area
    eq_qbasal = eq_Ptidal / A_surface * 1000  # mW/m²
    pole_qbasal = pole_Ptidal / A_surface * 1000
    print(f"{'Est. q_basal (mW/m²)':<25} {eq_qbasal:>20.1f} {pole_qbasal:>20.1f}")
    
    # Key insights
    print("\n--- KEY INSIGHTS ---")
    thickness_ratio = equator_results.cbe_km / pole_results.cbe_km if pole_results.cbe_km > 0 else float('inf')
    print(f"• Equator shell is {thickness_ratio:.1f}× thicker than pole")
    
    if equator_results.lid_fractions is not None:
        lid_diff = eq_lid - pole_lid
        print(f"• Equator has {abs(lid_diff):.1%} {'larger' if lid_diff > 0 else 'smaller'} lid fraction")
        
        conv_ratio = eq_Dconv / pole_Dconv if pole_Dconv > 0 else float('inf')
        print(f"• Equator D_conv is {conv_ratio:.1f}× {'larger' if conv_ratio > 1 else 'smaller'} than pole")
    
    print("=" * 70)


def run_comparison(n_iterations: int = 1000, 
                   n_workers: int | None = None,
                   seed: int = 42) -> Dict[str, MonteCarloResults]:
    """
    Run equator and pole simulations and compare results.
    
    Args:
        n_iterations: Number of Monte Carlo samples per region
        n_workers: Number of parallel workers
        seed: Random seed for reproducibility
    
    Returns:
        Dictionary with 'equator' and 'pole' MonteCarloResults
    """
    n_workers = default_worker_count() if n_workers is None else n_workers

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
    # Equator: reject subcritical → D_cond plateaus at ~10-15 km
    equator_config = SolverConfig(**base_kwargs, reject_subcritical=True)
    # Pole: keep subcritical as conductive → thin shells fully represented
    pole_config = SolverConfig(**base_kwargs, reject_subcritical=False)
    
    results = {}
    
    # =========================================================================
    # RUN EQUATOR SIMULATION
    # =========================================================================
    print("\n" + "=" * 60)
    print("  RUNNING EQUATOR SIMULATION")
    print("  (T_surf ~ 108 K, low tidal strain, low basal flux)")
    print("  reject_subcritical = True")
    print("=" * 60)
    
    equator_runner = MonteCarloRunner(
        n_iterations=n_iterations,
        seed=seed,
        verbose=True,
        n_workers=n_workers,
        config=equator_config,
        sampler_class=EquatorParameterSampler,
    )
    results['equator'] = equator_runner.run()
    
    # =========================================================================
    # RUN POLE SIMULATION
    # =========================================================================
    print("\n" + "=" * 60)
    print("  RUNNING POLE SIMULATION")
    print("  (T_surf ~ 50 K, high tidal strain, high basal flux)")
    print("  reject_subcritical = False")
    print("=" * 60)
    
    pole_runner = MonteCarloRunner(
        n_iterations=n_iterations,
        seed=seed + 10000,  # Different seed
        verbose=True,
        n_workers=n_workers,
        config=pole_config,
        sampler_class=PoleParameterSampler,
    )
    results['pole'] = pole_runner.run()
    
    return results


if __name__ == "__main__":
    mp.freeze_support()
    
    # Configuration
    N_ITERATIONS = 5000  # Samples per region
    N_WORKERS = default_worker_count()
    SEED = 42
    
    print("=" * 60)
    print("  EUROPA REGIONAL ICE SHELL COMPARISON")
    print("  Equator vs Pole Monte Carlo Analysis")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Iterations per region: {N_ITERATIONS}")
    print(f"  Workers: {N_WORKERS}")
    print(f"  Seed: {SEED}")
    
    # Run both simulations
    results = run_comparison(
        n_iterations=N_ITERATIONS,
        n_workers=N_WORKERS,
        seed=SEED,
    )
    
    # Print comparison
    print_comparison(results['equator'], results['pole'])
    
    # Save individual results
    save_results(results['equator'], os.path.join(RESULTS_DIR, "equator_results.npz"))
    save_results(results['pole'], os.path.join(RESULTS_DIR, "pole_results.npz"))
    
    # Save combined global results (equator + pole merged)
    eq = results['equator']
    po = results['pole']
    combined_path = os.path.join(RESULTS_DIR, "monte_carlo_results.npz")
    np.savez(combined_path,
             thicknesses_km=np.concatenate([eq.thicknesses_km, po.thicknesses_km]),
             D_cond_km=np.concatenate([eq.D_cond_km, po.D_cond_km]),
             D_conv_km=np.concatenate([eq.D_conv_km, po.D_conv_km]),
             lid_fractions=np.concatenate([eq.lid_fractions, po.lid_fractions]),
             Ra_values=np.concatenate([eq.Ra_values, po.Ra_values]),
             Nu_values=np.concatenate([eq.Nu_values, po.Nu_values]))
    
    n_combined = eq.n_valid + po.n_valid
    print(f"\nResults saved to:")
    print(f"  - results/equator_results.npz  ({eq.n_valid:,} samples)")
    print(f"  - results/pole_results.npz     ({po.n_valid:,} samples)")
    print(f"  - results/monte_carlo_results.npz  ({n_combined:,} combined)")
