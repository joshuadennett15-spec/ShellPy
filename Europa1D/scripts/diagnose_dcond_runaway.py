"""
Diagnostic script: Investigate conductive thickness runaway in >60 km shells.

Identifies which parameter combinations produce thick shells with subcritical
convective sublayers, causing the D_cond runaway observed in the shell structure plot.

Usage:
    python diagnose_dcond_runaway.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
RA_CRIT = 1.0e3


def load_data(filename="monte_carlo_results.npz"):
    data = np.load(os.path.join(RESULTS_DIR, filename))
    out = {
        'H': data["thicknesses_km"],
        'D_cond': data["D_cond_km"],
        'D_conv': data["D_conv_km"],
        'lid_frac': data["lid_fractions"],
    }
    if "Ra_values" in data:
        out['Ra'] = data["Ra_values"]
        out['subcrit'] = data["Ra_values"] < RA_CRIT
    else:
        out['Ra'] = np.zeros(len(out['H']))
        out['subcrit'] = np.zeros(len(out['H']), dtype=bool)

    # Sampled parameters
    for key in data.keys():
        if key.startswith('param_'):
            out[key[6:]] = data[key]

    return out


def main():
    d = load_data()
    N = len(d['H'])
    n_sub = d['subcrit'].sum()

    print("=" * 65)
    print("DIAGNOSTIC: Conductive Thickness Runaway Analysis")
    print("=" * 65)
    print(f"Total valid samples: {N}")
    print(f"Subcritical (Ra < {RA_CRIT:.0e}): {n_sub} ({100*n_sub/N:.1f}%)")
    print()

    # Split by total thickness
    mask_thin = d['H'] < 30
    mask_mid = (d['H'] >= 30) & (d['H'] < 60)
    mask_thick = d['H'] >= 60

    for label, mask in [("H < 30 km", mask_thin),
                        ("30-60 km", mask_mid),
                        ("H >= 60 km", mask_thick)]:
        n = mask.sum()
        if n == 0:
            continue
        n_s = d['subcrit'][mask].sum()
        print(f"  {label}: N={n}, subcritical={n_s} ({100*n_s/n:.1f}%)")
        print(f"    D_cond mean: {d['D_cond'][mask].mean():.1f} km, "
              f"D_conv mean: {d['D_conv'][mask].mean():.1f} km")
        print(f"    Ra: median={np.median(d['Ra'][mask]):.2e}, "
              f"min={d['Ra'][mask].min():.2e}, max={d['Ra'][mask].max():.2e}")
        print(f"    lid_frac: mean={d['lid_frac'][mask].mean():.2f}")
        if 'Q_v' in d:
            print(f"    Q_v: mean={d['Q_v'][mask].mean()/1e3:.1f} kJ/mol, "
                  f"std={d['Q_v'][mask].std()/1e3:.1f}")
        if 'T_surf' in d:
            print(f"    T_surf: mean={d['T_surf'][mask].mean():.1f} K, "
                  f"std={d['T_surf'][mask].std():.1f}")
        if 'd_grain' in d:
            print(f"    d_grain: median={np.median(d['d_grain'][mask])*1e3:.3f} mm")
        if 'epsilon_0' in d:
            print(f"    epsilon_0: median={np.median(d['epsilon_0'][mask]):.2e}")
        print()

    # =========================================================================
    # FIGURE: 4-panel diagnostic
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    sc = d['subcrit']
    sup = ~sc

    # (a) Ra vs H_total, colored by subcritical
    ax = axes[0, 0]
    ax.scatter(d['H'][sup], d['Ra'][sup], s=3, alpha=0.3, color='#4C72B0', label='Supercritical')
    ax.scatter(d['H'][sc], d['Ra'][sc], s=6, alpha=0.6, color='#C44E52', label='Subcritical', zorder=5)
    ax.axhline(RA_CRIT, color='orange', ls='--', lw=1.5, label=rf'$Ra_{{crit}}={RA_CRIT:.0e}$')
    ax.set_yscale('log')
    ax.set_xlabel("Total shell thickness (km)")
    ax.set_ylabel("Rayleigh number")
    ax.set_title("(a) Ra vs Total Thickness")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.2)

    # (b) D_conv vs H, colored by subcritical
    ax = axes[0, 1]
    ax.scatter(d['H'][sup], d['D_conv'][sup], s=3, alpha=0.3, color='#4C72B0', label='Supercritical')
    ax.scatter(d['H'][sc], d['D_conv'][sc], s=6, alpha=0.6, color='#C44E52', label='Subcritical', zorder=5)
    ax.plot([0, 100], [0, 100], 'k--', alpha=0.2, lw=0.8)
    ax.set_xlabel("Total shell thickness (km)")
    ax.set_ylabel("Convective sublayer D_conv (km)")
    ax.set_title("(b) D_conv vs Total Thickness")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.2)

    # (c) lid_fraction vs H, colored by subcritical
    ax = axes[1, 0]
    ax.scatter(d['H'][sup], d['lid_frac'][sup], s=3, alpha=0.3, color='#4C72B0', label='Supercritical')
    ax.scatter(d['H'][sc], d['lid_frac'][sc], s=6, alpha=0.6, color='#C44E52', label='Subcritical', zorder=5)
    ax.set_xlabel("Total shell thickness (km)")
    ax.set_ylabel(r"Lid fraction ($D_{cond}/H$)")
    ax.set_title("(c) Lid Fraction vs Total Thickness")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.2)

    # (d) Key parameter comparison: subcritical vs supercritical
    ax = axes[1, 1]
    params_to_compare = []
    if 'Q_v' in d:
        params_to_compare.append(('Q_v', d['Q_v'] / 1e3, 'kJ/mol'))
    if 'T_surf' in d:
        params_to_compare.append(('T_surf', d['T_surf'], 'K'))
    if 'd_grain' in d:
        params_to_compare.append(('d_grain', d['d_grain'] * 1e3, 'mm'))
    if 'epsilon_0' in d:
        params_to_compare.append(('eps_0', np.log10(d['epsilon_0']), 'log10'))

    if params_to_compare and n_sub > 0:
        labels = []
        positions = []
        box_data = []
        pos = 1
        for name, vals, unit in params_to_compare:
            box_data.extend([vals[sup], vals[sc]])
            positions.extend([pos, pos + 1])
            labels.extend([f"{name}\nSuper", f"{name}\nSub"])
            pos += 3

        bp = ax.boxplot(box_data, positions=positions, widths=0.7,
                        patch_artist=True, showfliers=False,
                        medianprops=dict(color="black", lw=1.5))
        colors = ['#4C72B0', '#C44E52'] * len(params_to_compare)
        for i, patch in enumerate(bp["boxes"]):
            patch.set_facecolor(colors[i])
            patch.set_alpha(0.7)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_title("(d) Parameter Distributions: Super vs Sub")
        ax.grid(True, alpha=0.2, axis='y')
    else:
        ax.text(0.5, 0.5, "No subcritical samples\nor no param data",
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title("(d) Parameter Distributions")

    fig.suptitle(f"Conductive Runaway Diagnostics (N={N:,}, subcrit={n_sub})",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, "dcond_runaway_diagnostic.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved diagnostic figure to: {out}")


if __name__ == "__main__":
    main()
