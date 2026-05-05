"""
Soderlund et al. 2013 Regional Comparison: Equator vs Pole

Ocean circulation pattern where heat is transported TO THE EQUATOR:
- EQUATOR: HIGH basal heat flux (~30-50 mW/m²) → THINNER shell
- POLE: LOW basal heat flux (~5-10 mW/m²) → THICKER shell

This is OPPOSITE to the default configuration where poles receive more heat.

Reference: Soderlund, K. M., et al. (2013). Ocean-driven heating of Europa's 
           icy shell at low latitudes. Nature Geoscience.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from runtime_support import configure_numeric_runtime, default_worker_count

configure_numeric_runtime()

import numpy as np
import multiprocessing as mp
from typing import Dict

from Monte_Carlo import MonteCarloRunner, MonteCarloResults, SolverConfig, save_results
from regional_samplers import SoderlundEquatorSampler, SoderlundPoleSampler

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')


def print_comparison(equator_results: MonteCarloResults, 
                     pole_results: MonteCarloResults) -> None:
    """Print side-by-side comparison of equator vs pole results."""
    
    print("\n" + "=" * 75)
    print("    SODERLUND ET AL. 2013: EQUATOR vs POLE ICE SHELL COMPARISON")
    print("    (Ocean circulation brings heat TO equator, away from poles)")
    print("=" * 75)
    
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
    A_surface = 4 * np.pi * (1.561e6) ** 2
    eq_qbasal = eq_Ptidal / A_surface * 1000
    pole_qbasal = pole_Ptidal / A_surface * 1000
    print(f"{'Est. q_basal (mW/m²)':<25} {eq_qbasal:>20.1f} {pole_qbasal:>20.1f}")
    
    # Key insights
    print("\n--- KEY INSIGHTS (Soderlund 2013 Pattern) ---")
    if pole_results.cbe_km > 0:
        thickness_ratio = pole_results.cbe_km / equator_results.cbe_km
        print(f"• Pole shell is {thickness_ratio:.1f}× thicker than equator")
    
    if equator_results.lid_fractions is not None:
        lid_diff = pole_lid - eq_lid
        print(f"• Pole has {abs(lid_diff):.1%} {'larger' if lid_diff > 0 else 'smaller'} lid fraction")
        
        if pole_Dconv > 0 and eq_Dconv > 0:
            conv_ratio = eq_Dconv / pole_Dconv
            print(f"• Equator D_conv is {conv_ratio:.1f}× {'larger' if conv_ratio > 1 else 'smaller'} than pole")
    
    print("\n• Interpretation: High basal heat at equator thins the shell there,")
    print("  while cold poles with low basal flux grow thick conductive shells.")
    print("=" * 75)


def run_soderlund_comparison(n_iterations: int = 1000, 
                              n_workers: int | None = None,
                              seed: int = 42) -> Dict[str, MonteCarloResults]:
    """
    Run equator and pole simulations with Soderlund 2013 configuration.
    """
    n_workers = default_worker_count() if n_workers is None else n_workers

    config = SolverConfig(
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
    
    results = {}
    
    # =========================================================================
    # RUN EQUATOR SIMULATION (HIGH basal flux)
    # =========================================================================
    print("\n" + "=" * 60)
    print("  RUNNING SODERLUND EQUATOR SIMULATION")
    print("  (T_surf ~ 108 K, HIGH basal flux ~30-50 mW/m²)")
    print("=" * 60)
    
    equator_runner = MonteCarloRunner(
        n_iterations=n_iterations,
        seed=seed,
        verbose=True,
        n_workers=n_workers,
        config=config,
        sampler_class=SoderlundEquatorSampler,
    )
    results['equator'] = equator_runner.run()
    
    # =========================================================================
    # RUN POLE SIMULATION (LOW basal flux)
    # =========================================================================
    print("\n" + "=" * 60)
    print("  RUNNING SODERLUND POLE SIMULATION")
    print("  (T_surf ~ 50 K, LOW basal flux ~5-10 mW/m²)")
    print("=" * 60)
    
    pole_runner = MonteCarloRunner(
        n_iterations=n_iterations,
        seed=seed + 10000,
        verbose=True,
        n_workers=n_workers,
        config=config,
        sampler_class=SoderlundPoleSampler,
    )
    results['pole'] = pole_runner.run()
    
    return results


if __name__ == "__main__":
    mp.freeze_support()
    
    # Configuration
    N_ITERATIONS = 1000
    N_WORKERS = default_worker_count()
    SEED = 42
    
    print("=" * 60)
    print("  SODERLUND ET AL. 2013 REGIONAL COMPARISON")
    print("  Ocean Heat Transport TO Equator (opposite of default)")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Iterations per region: {N_ITERATIONS}")
    print(f"  Workers: {N_WORKERS}")
    print(f"  Seed: {SEED}")
    print(f"\nBasal Heat Flux Pattern:")
    print(f"  EQUATOR: ~30-50 mW/m² (HIGH)")
    print(f"  POLE:    ~5-10 mW/m²  (LOW)")
    
    # Run both simulations
    results = run_soderlund_comparison(
        n_iterations=N_ITERATIONS,
        n_workers=N_WORKERS,
        seed=SEED,
    )
    
    # Print comparison
    print_comparison(results['equator'], results['pole'])
    
    # Save results
    save_results(results['equator'], os.path.join(RESULTS_DIR, "soderlund_equator_results.npz"))
    save_results(results['pole'], os.path.join(RESULTS_DIR, "soderlund_pole_results.npz"))
    
    print("\nResults saved to:")
    print("  - results/soderlund_equator_results.npz")
    print("  - results/soderlund_pole_results.npz")
