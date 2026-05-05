#!/usr/bin/env python3
"""
Single broad-prior Monte Carlo archive at 35 deg latitude, uniform transport.

Produces a clean N=25000 ensemble from UNCONSTRAINED priors for use in the
Bayesian identifiability analysis (restricted-update ladder, covariance
eigenanalysis, Shapley effects).  Does NOT iterate or tighten priors.

The output NPZ contains all raw param_* arrays plus D_cond_km and
thicknesses_km, ready for post-hoc importance reweighting against
Juno D_cond = 29 +/- 10 km.

Usage:
    python run_midlat_broad.py                   # N=25000 (default)
    python run_midlat_broad.py -n 15000          # smaller run
    python run_midlat_broad.py --seed 12345      # reproducible
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import argparse
import json
import time
import multiprocessing as mp

from runtime_support import configure_numeric_runtime, resolve_worker_count

configure_numeric_runtime()

import numpy as np

from Monte_Carlo import MonteCarloRunner, SolverConfig, save_results
from juno_constrained_sampler import JunoConstrainedMidLatSampler
from constants import Rheology

# ═══════════════════════════════════════════════════════════════════════════
# Physical constants at 35 deg latitude
# ═══════════════════════════════════════════════════════════════════════════

PHI_DEG = 35.0
PHI_RAD = np.radians(PHI_DEG)

# Surface temperature: Ashkenazy (2019) cos^p interpolation
T_EQ = 110.0   # K
T_FLOOR = 46.0  # K
_SURF_EXP = 1.25
_cos_p = np.cos(PHI_RAD) ** _SURF_EXP
T_SURF_35 = float(((T_EQ**4 - T_FLOOR**4) * _cos_p + T_FLOOR**4) ** 0.25)

# Tidal strain: Beuthe (2013) eccentricity-tide pattern
EPS_EQ = 6e-6
EPS_POLE = 1.2e-5
_c_strain = (EPS_POLE / EPS_EQ)**2 - 1.0
EPS_35 = float(EPS_EQ * np.sqrt(1 + _c_strain * np.sin(PHI_RAD)**2))

# Broad (unconstrained) priors — matches AuditedShellSampler defaults
BROAD_PRIORS = {
    'q_basal_lo': 0.005,       # 5 mW/m^2
    'q_basal_hi': 0.025,       # 25 mW/m^2
    'd_grain_log_center': float(np.log10(6e-4)),   # 0.6 mm
    'd_grain_log_sigma': 0.35,
    'd_grain_lo': 5e-5,        # 0.05 mm
    'd_grain_hi': 3e-3,        # 3.0 mm
    'T_surf_mean': T_SURF_35,
    'T_surf_std': 5.0,
    'T_surf_clip': [85.0, 115.0],
    'eps_log_center': float(np.log10(EPS_35)),
    'eps_log_sigma': 0.2,
    'eps_clip': [2e-6, 3e-5],
    'q_tidal_mult': 1.0,       # uniform transport
}

# Directories
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'midlat_juno')
PRIORS_FILE = os.path.join(RESULTS_DIR, 'current_priors.json')


def write_priors_json(priors, filepath):
    """Write prior parameters to JSON for the constrained sampler."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    safe = {}
    for k, v in priors.items():
        if isinstance(v, (np.floating, np.integer)):
            safe[k] = float(v)
        elif isinstance(v, np.ndarray):
            safe[k] = v.tolist()
        elif isinstance(v, list):
            safe[k] = [float(x) if isinstance(x, (np.floating, np.integer)) else x
                        for x in v]
        else:
            safe[k] = v
    with open(filepath, 'w') as f:
        json.dump(safe, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Single broad-prior MC archive at 35 deg, uniform transport")
    parser.add_argument("-n", type=int, default=20000,
                        help="Number of MC iterations (default 20000)")
    parser.add_argument("--seed", type=int, default=35042,
                        help="Random seed (default 35042)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Parallel workers (default: shared Windows-safe default)")
    args = parser.parse_args()

    n_workers = resolve_worker_count(args.workers)

    assert Rheology.MODEL == "Andrade", f"Expected Andrade rheology, got {Rheology.MODEL}"

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Write broad (unconstrained) priors to JSON
    print("Writing broad priors to current_priors.json ...")
    write_priors_json(BROAD_PRIORS, PRIORS_FILE)
    print(f"  q_basal: [{BROAD_PRIORS['q_basal_lo']*1e3:.1f}, "
          f"{BROAD_PRIORS['q_basal_hi']*1e3:.1f}] mW/m^2")
    print(f"  d_grain: center={10**BROAD_PRIORS['d_grain_log_center']*1e3:.2f} mm, "
          f"sigma={BROAD_PRIORS['d_grain_log_sigma']:.2f} dex")
    print(f"  T_surf(35 deg) = {T_SURF_35:.1f} K")
    print(f"  epsilon_0(35 deg) = {EPS_35:.3e}")
    print(f"  q_tidal_mult = {BROAD_PRIORS['q_tidal_mult']:.4f} (uniform)")

    # Run MC
    config = SolverConfig(reject_subcritical=False)
    runner = MonteCarloRunner(
        n_iterations=args.n,
        seed=args.seed,
        verbose=True,
        n_workers=n_workers,
        config=config,
        sampler_class=JunoConstrainedMidLatSampler,
    )

    print(f"\n{'='*60}")
    print(f"  MC RUN: midlat35_uniform_broad  (N={args.n})")
    print(f"  Workers: {n_workers}")
    print(f"{'='*60}")

    t0 = time.time()
    results = runner.run()
    elapsed = time.time() - t0

    # Save
    outpath = os.path.join(RESULTS_DIR, "midlat35_uniform_broad.npz")
    save_results(results, outpath)

    n_valid = results.n_valid
    print(f"\n{'='*60}")
    print(f"  COMPLETE")
    print(f"  Valid samples: {n_valid} / {args.n}")
    print(f"  Output: {os.path.abspath(outpath)}")
    print(f"  Runtime: {elapsed/60:.1f} min")

    # Quick D_cond summary
    if results.D_cond_km is not None:
        dc = results.D_cond_km
        print(f"  D_cond: median={np.median(dc):.1f} km, "
              f"68%=[{np.percentile(dc, 15.87):.1f}, {np.percentile(dc, 84.13):.1f}] km")
    if results.thicknesses_km is not None:
        ht = results.thicknesses_km
        print(f"  H_total: median={np.median(ht):.1f} km, "
              f"68%=[{np.percentile(ht, 15.87):.1f}, {np.percentile(ht, 84.13):.1f}] km")

    print(f"{'='*60}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
