"""
Quick 500-sample MC run with the 2026 tightened priors.
Prints rejection stats and shell thickness distribution.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from runtime_support import configure_numeric_runtime

configure_numeric_runtime()

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from Monte_Carlo import MonteCarloRunner, SolverConfig
from audited_sampler import AuditedShellSampler

def main():
    N = 500
    SEED = 2026

    config = SolverConfig(reject_subcritical=False)

    runner = MonteCarloRunner(
        n_iterations=N,
        seed=SEED,
        verbose=True,
        config=config,
        sampler_class=AuditedShellSampler,
    )

    results = runner.run()

    # --- Summary stats ---
    n_rejected = N - results.n_valid
    pct_rejected = 100.0 * n_rejected / N

    print("\n" + "=" * 60)
    print(f"  NEW PRIORS — {N} samples, seed={SEED}")
    print("=" * 60)
    print(f"  Valid:    {results.n_valid}/{N}  ({100 - pct_rejected:.1f}%)")
    print(f"  Rejected: {n_rejected}/{N}  ({pct_rejected:.1f}%)")
    print(f"  Runtime:  {results.runtime_seconds:.1f} s")
    print()
    print(f"  CBE (mode):  {results.cbe_km:.1f} km")
    print(f"  Median:      {results.median_km:.1f} km")
    print(f"  Mean:        {results.mean_km:.1f} km")
    print(f"  1-sigma:    [{results.sigma_1_low_km:.1f}, {results.sigma_1_high_km:.1f}] km")
    print(f"  Full range:  [{results.thicknesses_km.min():.1f}, {results.thicknesses_km.max():.1f}] km")

    # --- Convection breakdown ---
    if results.D_conv_km is not None:
        n_conductive_only = np.sum(results.D_conv_km < 0.5)
        n_convecting = results.n_valid - n_conductive_only
        print()
        print(f"  Convecting:       {n_convecting}/{results.n_valid} ({100*n_convecting/results.n_valid:.1f}%)")
        print(f"  Conductive-only:  {n_conductive_only}/{results.n_valid} ({100*n_conductive_only/results.n_valid:.1f}%)")

        if results.lid_fractions is not None:
            print(f"  Lid fraction:     median={np.median(results.lid_fractions):.2f}, "
                  f"mean={np.mean(results.lid_fractions):.2f}")

    # --- Key sampled param summaries ---
    if results.sampled_params:
        print()
        print("  Sampled parameter medians:")
        for key in ['T_surf', 'Q_v', 'Q_b', 'epsilon_0', 'd_grain']:
            if key in results.sampled_params:
                arr = results.sampled_params[key]
                if key == 'epsilon_0':
                    print(f"    {key:12s}: {np.median(arr):.2e}  [{np.min(arr):.2e}, {np.max(arr):.2e}]")
                elif key == 'd_grain':
                    print(f"    {key:12s}: {np.median(arr)*1e3:.3f} mm  [{np.min(arr)*1e3:.3f}, {np.max(arr)*1e3:.3f}] mm")
                elif key in ('Q_v', 'Q_b'):
                    print(f"    {key:12s}: {np.median(arr)/1e3:.1f} kJ/mol  [{np.min(arr)/1e3:.1f}, {np.max(arr)/1e3:.1f}]")
                else:
                    print(f"    {key:12s}: {np.median(arr):.1f}  [{np.min(arr):.1f}, {np.max(arr):.1f}]")

    # --- Figure ---
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)

    # Panel 1: Shell thickness histogram
    ax = axes[0]
    ax.hist(results.thicknesses_km, bins=30, color='steelblue', edgecolor='white', alpha=0.85)
    ax.axvline(results.median_km, color='k', ls='--', lw=1.5, label=f'median={results.median_km:.1f} km')
    ax.axvline(results.cbe_km, color='crimson', ls='-', lw=1.5, label=f'CBE={results.cbe_km:.1f} km')
    ax.set_xlabel('Shell thickness (km)')
    ax.set_ylabel('Count')
    ax.set_title(f'New priors — {results.n_valid}/{N} valid')
    ax.legend(fontsize=8)

    # Panel 2: D_cond vs D_conv
    ax = axes[1]
    if results.D_cond_km is not None and results.D_conv_km is not None:
        mask_conv = results.D_conv_km >= 0.5
        ax.scatter(results.D_cond_km[mask_conv], results.D_conv_km[mask_conv],
                   s=8, alpha=0.5, c='steelblue', label='convecting')
        ax.scatter(results.D_cond_km[~mask_conv], results.D_conv_km[~mask_conv],
                   s=8, alpha=0.5, c='coral', label='conductive-only')
        ax.set_xlabel('D_cond (km)')
        ax.set_ylabel('D_conv (km)')
        ax.set_title('Lid vs convective layer')
        ax.legend(fontsize=8)

    # Panel 3: Lid fraction histogram
    ax = axes[2]
    if results.lid_fractions is not None:
        ax.hist(results.lid_fractions, bins=30, color='goldenrod', edgecolor='white', alpha=0.85)
        ax.set_xlabel('Lid fraction (D_cond / H)')
        ax.set_ylabel('Count')
        ax.set_title('Conductive lid fraction')

    outpath = os.path.join(os.path.dirname(__file__), '..', 'figures', 'new_priors_500.png')
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    fig.savefig(outpath, dpi=150)
    print(f"\n  Figure saved to: {os.path.abspath(outpath)}")


if __name__ == '__main__':
    main()
