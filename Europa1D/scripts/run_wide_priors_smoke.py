"""
N=500 prior-predictive smoke test for AuditedWidePriorsSampler.

Purpose: check that the literature-envelope wide priors produce physically
reasonable D_cond, D_conv, H_total, convective fraction, and acceptable
invalid/subcritical rates BEFORE committing to a full N=15k rerun.

This is a sensitivity test against the audited baseline (mc_15000_optionA_v2_andrade),
not a replacement for it. Output goes to results/mc_500_wide_priors_andrade.npz.
"""
import os
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(THIS_DIR, "..", "src")
RESULTS_DIR = os.path.join(THIS_DIR, "..", "results")
sys.path.insert(0, SRC_DIR)

import numpy as np

from Monte_Carlo import MonteCarloRunner, SolverConfig, save_results
from audited_wide_priors_sampler import AuditedWidePriorsSampler


N_SAMPLES = 500
SEED = 42
OUTPUT_PATH = os.path.join(RESULTS_DIR, "mc_500_wide_priors_andrade.npz")


def _summarize(results):
    """Print prior-predictive diagnostics."""
    print(f"\n{'=' * 60}")
    print("WIDE LITERATURE-ENVELOPE PRIOR — N=500 SMOKE")
    print(f"{'=' * 60}")
    print(f"  Valid:        {results.n_valid}/{results.n_iterations}"
          f"  ({100 * results.n_valid / results.n_iterations:.1f}%)")
    print(f"  D_cond CBE:   {results.cbe_km:.1f} km")
    print(f"  D_cond med:   {results.median_km:.1f} km")
    print(f"  D_cond [16,84]: [{results.sigma_1_low_km:.1f}, "
          f"{results.sigma_1_high_km:.1f}] km")
    print(f"  Runtime:      {results.runtime_seconds:.0f}s")

    archive = np.load(OUTPUT_PATH)
    keys = archive.files
    if "D_conv_km" in keys:
        d_conv = archive["D_conv_km"]
        d_conv = d_conv[np.isfinite(d_conv)]
        if d_conv.size:
            print(f"  D_conv med:   {np.median(d_conv):.1f} km  "
                  f"({np.percentile(d_conv, 16):.1f}, "
                  f"{np.percentile(d_conv, 84):.1f})")
    if "thicknesses_km" in keys:
        h = archive["thicknesses_km"]
        h = h[np.isfinite(h)]
        if h.size:
            print(f"  H_total med:  {np.median(h):.1f} km  "
                  f"({np.percentile(h, 16):.1f}, "
                  f"{np.percentile(h, 84):.1f})")
    if "convective_flag" in keys:
        flag = archive["convective_flag"]
        if flag.size:
            print(f"  Convecting:   {100 * np.mean(flag):.1f}%")
    if "Ra_values" in keys and "param_d_grain" in keys:
        print(f"\n  Param ranges (drawn):")
        for key, scale, label, unit in [
            ("param_d_grain", 1e3, "d_grain", "mm"),
            ("param_epsilon_0", 1.0, "epsilon_0", "s^-1"),
            ("param_Q_v", 1e-3, "Q_v", "kJ/mol"),
            ("param_Q_b", 1e-3, "Q_b", "kJ/mol"),
            ("param_f_porosity", 1.0, "f_porosity", ""),
        ]:
            if key in keys:
                v = archive[key] * scale
                print(f"    {label:12s} [{np.min(v):.3g}, {np.max(v):.3g}] "
                      f"med={np.median(v):.3g} {unit}")


def main():
    print(f"Running N={N_SAMPLES} wide-prior smoke (seed={SEED})")
    config = SolverConfig(reject_subcritical=False)
    runner = MonteCarloRunner(
        n_iterations=N_SAMPLES,
        seed=SEED,
        verbose=True,
        config=config,
        sampler_class=AuditedWidePriorsSampler,
    )
    results = runner.run()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    save_results(results, OUTPUT_PATH)
    _summarize(results)
    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
