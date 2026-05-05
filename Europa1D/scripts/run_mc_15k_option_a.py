#!/usr/bin/env python3
"""
Option A: Shell-level Monte Carlo with audited 2026 priors.

Key changes from default Howell sampler (per PARAMETER_PRIOR_AUDIT_2026.md):
  - q_basal sampled directly: U(10, 30) mW/m^2
    (Levin et al. 2026; Tobie et al. 2003; chondritic baseline ~10 mW/m^2)
  - f_salt = 0, B_k = 1 (pure-ice baseline; salinity is a scenario, not a prior)
  - epsilon_0 clip tightened to [2e-6, 3.4e-5]
  - f_porosity narrowed to [0, 0.10]
  - T_surf clip tightened to [80, 120] K
  - All other params: Howell (2021) Table 1

Runs two ensembles:
  1. Maxwell rheology (15,000 iterations)
  2. Andrade rheology (15,000 iterations)

References:
  Howell (2021)           doi:10.3847/PSJ/abfe10
  Levin et al. (2026)     doi:10.1038/s41550-025-02718-0
  Tobie et al. (2003)     doi:10.1029/2003JE002099
  McCarthy & Cooper (2016) doi:10.1016/j.epsl.2016.03.006
  Hussmann & Spohn (2004) Laplace resonance stability
"""
import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from runtime_support import configure_numeric_runtime

configure_numeric_runtime()

import numpy as np
import multiprocessing as mp

from Monte_Carlo import (
    MonteCarloRunner, SolverConfig, HowellParameterSampler, save_results,
)
from constants import Planetary, Thermal

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'src', 'config.json')


class AuditedShellSampler(HowellParameterSampler):
    """
    2026-audited parameter sampler for shell-level thickness study.

    Primary change: samples q_basal directly as a shell-level forcing
    prior (U(10, 30) mW/m^2) instead of deriving it from P_tidal.
    P_tidal is back-calculated so the worker function is unmodified.

    Removes f_salt and B_k from the active prior (pure-ice baseline).
    Tightens epsilon_0 and f_porosity bounds.
    """

    # Shell-level basal flux prior (mW/m^2)
    Q_BASAL_LO = 0.010   # 10 mW/m^2
    Q_BASAL_HI = 0.030   # 30 mW/m^2

    def sample(self):
        params = super().sample()

        # ── 1. Sample q_basal directly ──────────────────────────────────
        q_basal_target = self.rng.uniform(self.Q_BASAL_LO, self.Q_BASAL_HI)

        # Compute radiogenic contribution from already-sampled H_rad & D_H2O
        H_rad = params['H_rad']
        D_H2O = params['D_H2O']
        R_rock = Planetary.RADIUS - D_H2O
        M_rock = (4.0 / 3.0) * np.pi * (R_rock ** 3) * 3500.0
        q_radiogenic = (H_rad * M_rock) / Planetary.AREA

        # Back-calculate P_tidal so the worker sees the right q_basal
        q_silicate_tidal = max(0.0, q_basal_target - q_radiogenic)
        params['P_tidal'] = q_silicate_tidal * Planetary.AREA

        # Store q_basal for diagnostics
        params['q_basal'] = q_basal_target

        # ── 2. Pure-ice baseline ────────────────────────────────────────
        params['f_salt'] = 0.0
        params['B_k'] = 1.0

        # ── 3. Tighten epsilon_0 clip ───────────────────────────────────
        eps = params['epsilon_0']
        if eps < 2e-6 or eps > 3.4e-5:
            # Re-draw within tighter bounds
            while True:
                eps = 10 ** self.rng.normal(np.log10(1.2e-5), 0.3)
                if 2e-6 <= eps <= 3.4e-5:
                    break
            params['epsilon_0'] = eps

        # ── 4. Narrow f_porosity ────────────────────────────────────────
        params['f_porosity'] = self.rng.uniform(0.0, 0.10)

        # ── 5. Tighten T_surf clip ──────────────────────────────────────
        T = params['T_surf']
        if T < 80.0 or T > 120.0:
            params['T_surf'] = np.clip(
                self.rng.normal(104.0, 7.0), 80.0, 120.0
            )

        # ── 6. Truncate H_rad positive ──────────────────────────────────
        if params['H_rad'] <= 0:
            params['H_rad'] = abs(self.rng.normal(4.5e-12, 1.0e-12))

        return params


def _set_rheology(model_name: str):
    """Update config.json rheology model (Maxwell or Andrade)."""
    with open(CONFIG_PATH, 'r') as f:
        cfg = json.load(f)
    cfg['rheology']['model'] = model_name
    with open(CONFIG_PATH, 'w') as f:
        json.dump(cfg, f, indent=4)


def _run_ensemble(model_name, n_iterations, seed):
    """Run one MC ensemble and return the output path."""
    print(f"\n{'=' * 60}")
    print(f"ENSEMBLE: {model_name} rheology")
    print(f"  q_basal ~ U(10, 30) mW/m^2 (shell-level prior)")
    print(f"  Pure-ice baseline (f_salt=0, B_k=1)")
    print(f"  {n_iterations:,} iterations")
    print(f"{'=' * 60}")

    config = SolverConfig(reject_subcritical=False)

    runner = MonteCarloRunner(
        n_iterations=n_iterations,
        seed=seed,
        verbose=True,
        config=config,
        sampler_class=AuditedShellSampler,
    )
    results = runner.run()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    tag = model_name.lower()
    output_path = os.path.join(
        RESULTS_DIR, f"mc_15000_optionA_{tag}.npz"
    )
    save_results(results, output_path)

    print(f"\n--- {model_name} RESULTS ---")
    print(f"  Valid:   {results.n_valid}/{results.n_iterations}")
    print(f"  CBE:     {results.cbe_km:.1f} km")
    print(f"  Median:  {results.median_km:.1f} km")
    print(f"  1-sigma: [{results.sigma_1_low_km:.1f}, {results.sigma_1_high_km:.1f}] km")
    print(f"  Runtime: {results.runtime_seconds:.0f}s")

    return output_path


def main():
    n_iterations = 15000

    # Save original config
    with open(CONFIG_PATH, 'r') as f:
        original_config = json.load(f)

    try:
        # Ensemble 1: Maxwell
        _set_rheology("Maxwell")
        # Force reload of constants by restarting import
        # (ConfigManager is a singleton, need fresh process)
        path_maxwell = _run_ensemble("Maxwell", n_iterations, seed=42)

        # Ensemble 2: Andrade
        # ConfigManager is a singleton loaded at import time.
        # We need a subprocess to pick up the new config.
        _set_rheology("Andrade")
        print("\n>>> Launching Andrade ensemble as subprocess...")

        import subprocess
        result = subprocess.run(
            [sys.executable, "-c", f"""
import sys, os
sys.path.insert(0, os.path.join(r'{os.path.dirname(__file__)}', '..', 'src'))

# Force fresh import with Andrade config
from Monte_Carlo import MonteCarloRunner, SolverConfig, save_results
# Import the sampler from this script's module
sys.path.insert(0, r'{os.path.dirname(__file__)}')

import numpy as np
from constants import Planetary, Thermal
from Monte_Carlo import HowellParameterSampler

class AuditedShellSampler(HowellParameterSampler):
    Q_BASAL_LO = 0.010
    Q_BASAL_HI = 0.030
    def sample(self):
        params = super().sample()
        q_basal_target = self.rng.uniform(self.Q_BASAL_LO, self.Q_BASAL_HI)
        H_rad = params['H_rad']
        D_H2O = params['D_H2O']
        R_rock = Planetary.RADIUS - D_H2O
        M_rock = (4.0 / 3.0) * np.pi * (R_rock ** 3) * 3500.0
        q_radiogenic = (H_rad * M_rock) / Planetary.AREA
        q_silicate_tidal = max(0.0, q_basal_target - q_radiogenic)
        params['P_tidal'] = q_silicate_tidal * Planetary.AREA
        params['q_basal'] = q_basal_target
        params['f_salt'] = 0.0
        params['B_k'] = 1.0
        eps = params['epsilon_0']
        if eps < 2e-6 or eps > 3.4e-5:
            while True:
                eps = 10 ** self.rng.normal(np.log10(1.2e-5), 0.3)
                if 2e-6 <= eps <= 3.4e-5:
                    break
            params['epsilon_0'] = eps
        params['f_porosity'] = self.rng.uniform(0.0, 0.10)
        T = params['T_surf']
        if T < 80.0 or T > 120.0:
            params['T_surf'] = np.clip(self.rng.normal(104.0, 7.0), 80.0, 120.0)
        if params['H_rad'] <= 0:
            params['H_rad'] = abs(self.rng.normal(4.5e-12, 1.0e-12))
        return params

import multiprocessing as mp
mp.freeze_support()
config = SolverConfig(reject_subcritical=False)
runner = MonteCarloRunner(
    n_iterations={n_iterations}, seed=10042, verbose=True,
    config=config, sampler_class=AuditedShellSampler,
)
results = runner.run()
save_results(results, r'{os.path.join(RESULTS_DIR, "mc_15000_optionA_andrade.npz")}')
print(f'CBE: {{results.cbe_km:.1f}} km')
print(f'Median: {{results.median_km:.1f}} km')
print(f'1-sigma: [{{results.sigma_1_low_km:.1f}}, {{results.sigma_1_high_km:.1f}}] km')
print(f'Valid: {{results.n_valid}}/{{results.n_iterations}}')
"""],
            timeout=7200,
        )
        if result.returncode != 0:
            print(f"Andrade subprocess failed: {result.stderr}")

    finally:
        # Restore original config
        with open(CONFIG_PATH, 'w') as f:
            json.dump(original_config, f, indent=4)
        print("\nRestored original config.json")

    print("\n" + "=" * 60)
    print("DONE — both ensembles saved:")
    print(f"  Maxwell: results/mc_15000_optionA_maxwell.npz")
    print(f"  Andrade: results/mc_15000_optionA_andrade.npz")
    print("=" * 60)


if __name__ == "__main__":
    mp.freeze_support()
    main()
