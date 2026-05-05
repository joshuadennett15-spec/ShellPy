"""
Improved shell structure plot for all ocean heat transport scenarios.

Left column: D_cond and D_conv distributions (KDE), split by subpopulation.
Right column: Stacked area showing mean D_cond + D_conv vs total H,
              with convective/conductive boundary highlighted.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
RA_CRIT = 1000.0

SCENARIOS = [
    ('Global',              ['budget_global'],                                                '#333333'),
    ('Uniform',             ['budget_uniform_equator', 'budget_uniform_pole'],                '#1B9E77'),
    ('Equatorial-enhanced', ['budget_soderlund2014_equator', 'budget_soderlund2014_pole'],    '#D95F02'),
    ('Polar-enhanced',      ['budget_lemasquerier2023_equator', 'budget_lemasquerier2023_pole'], '#377EB8'),
]

C_LID = '#4A90D9'
C_CONV = '#E8645A'
C_LID_DARK = '#2C5F99'
C_CONV_DARK = '#B83B31'


def load_and_concat(file_list):
    all_H, all_Ra, all_Dc, all_Dv, all_lid = [], [], [], [], []
    for fname in file_list:
        d = np.load(os.path.join(RESULTS_DIR, f'{fname}.npz'))
        all_H.append(d['thicknesses_km'])
        all_Ra.append(d['Ra_values'])
        all_Dc.append(d['D_cond_km'])
        all_Dv.append(d['D_conv_km'])
        all_lid.append(d['lid_fractions'])
    return dict(
        H=np.concatenate(all_H), Ra=np.concatenate(all_Ra),
        Dc=np.concatenate(all_Dc), Dv=np.concatenate(all_Dv),
        lid=np.concatenate(all_lid),
    )


def safe_kde(data, x, bw_method=None):
    if len(data) < 5 or np.std(data) < 1e-10:
        return np.zeros_like(x)
    try:
        return gaussian_kde(data, bw_method=bw_method)(x)
    except np.linalg.LinAlgError:
        return np.zeros_like(x)


def binned_structure(H, Dc, Dv, n_bins=25):
    """Compute mean D_cond and D_conv in bins of total H."""
    edges = np.linspace(H.min() - 0.5, H.max() + 0.5, n_bins + 1)
    centres = (edges[:-1] + edges[1:]) / 2
    mean_dc = np.full(n_bins, np.nan)
    mean_dv = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=int)
    for i in range(n_bins):
        mask = (H >= edges[i]) & (H < edges[i + 1])
        counts[i] = mask.sum()
        if counts[i] >= 3:
            mean_dc[i] = np.mean(Dc[mask])
            mean_dv[i] = np.mean(Dv[mask])
    valid = counts >= 3
    return centres[valid], mean_dc[valid], mean_dv[valid], counts[valid]


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    fig, axes = plt.subplots(4, 2, figsize=(15, 18))
    fig.subplots_adjust(hspace=0.35, wspace=0.28)

    x_km = np.linspace(0, 100, 500)

    for row, (label, files, accent) in enumerate(SCENARIOS):
        data = load_and_concat(files)
        H, Dc, Dv, Ra = data['H'], data['Dc'], data['Dv'], data['Ra']
        m_conv = Ra >= RA_CRIT
        m_cond = ~m_conv
        n_conv, n_cond = m_conv.sum(), m_cond.sum()
        n_total = len(H)

        ax_dist = axes[row, 0]
        ax_stack = axes[row, 1]

        # ── Left: D_cond and D_conv distributions ────────────────────────────
        pdf_dc = safe_kde(Dc, x_km)
        pdf_dv = safe_kde(Dv[Dv > 0.5], x_km)

        ax_dist.fill_between(x_km, pdf_dc, alpha=0.30, color=C_LID, zorder=2)
        ax_dist.plot(x_km, pdf_dc, color=C_LID_DARK, lw=2, zorder=3,
                     label=r'$D_{\rm cond}$ (lid)')
        ax_dist.fill_between(x_km, pdf_dv, alpha=0.30, color=C_CONV, zorder=2)
        ax_dist.plot(x_km, pdf_dv, color=C_CONV_DARK, lw=2, zorder=3,
                     label=r'$D_{\rm conv}$ (sublayer)')

        med_dc = np.median(Dc)
        med_dv = np.median(Dv[Dv > 0.5]) if (Dv > 0.5).sum() > 0 else 0
        ax_dist.axvline(med_dc, color=C_LID_DARK, ls=':', lw=1.5, alpha=0.7)
        ax_dist.axvline(med_dv, color=C_CONV_DARK, ls=':', lw=1.5, alpha=0.7)

        pct_conv = 100 * n_conv / n_total
        stats = (f'N = {n_total:,}\n'
                 f'Convective: {n_conv:,} ({pct_conv:.0f}%)\n'
                 f'Conductive: {n_cond:,} ({100 - pct_conv:.0f}%)\n'
                 f'med $D_c$ = {med_dc:.1f} km\n'
                 f'med $D_v$ = {med_dv:.1f} km')
        ax_dist.text(0.97, 0.95, stats, transform=ax_dist.transAxes,
                     fontsize=8, va='top', ha='right', fontfamily='monospace',
                     bbox=dict(boxstyle='round,pad=0.4', fc='white', alpha=0.85, ec='#ccc'))

        ax_dist.set_xlim(0, 90)
        ax_dist.set_ylim(bottom=0)
        ax_dist.legend(fontsize=9, loc='upper left', framealpha=0.9)
        ax_dist.set_ylabel('Probability density', fontsize=10)
        ax_dist.set_title(f'({chr(97 + row * 2)})  {label} \u2014 layer distributions',
                          fontsize=11, fontweight='bold', loc='left')
        if row == 3:
            ax_dist.set_xlabel('Layer thickness (km)', fontsize=10)

        # ── Right: Stacked structure vs total H ──────────────────────────────
        centres, mean_dc, mean_dv, counts = binned_structure(H, Dc, Dv, n_bins=30)

        ax_stack.fill_between(centres, 0, mean_dc, color=C_LID, alpha=0.65,
                              label=r'$D_{\rm cond}$', zorder=2)
        ax_stack.fill_between(centres, mean_dc, mean_dc + mean_dv,
                              color=C_CONV, alpha=0.65,
                              label=r'$D_{\rm conv}$', zorder=2)

        ax_stack.plot([0, 100], [0, 100], 'k--', lw=1, alpha=0.3, zorder=1)
        ax_stack.plot(centres, mean_dc + mean_dv, 'k-', lw=2, alpha=0.8, zorder=4,
                      label=r'$H_{\rm total}$ (mean)')

        # Mark bins where lid fraction > 85% (conductive regime)
        for i, c in enumerate(centres):
            if mean_dv[i] < 0.15 * (mean_dc[i] + mean_dv[i]):
                ax_stack.plot(c, mean_dc[i] + mean_dv[i], 'x',
                              color='#555', ms=5, mew=1.5, zorder=5)

        # Shade convective / conductive regimes on x-axis
        if n_conv > 10 and n_cond > 10:
            h_conv_range = (np.percentile(H[m_conv], 5), np.percentile(H[m_conv], 95))
            h_cond_range = (np.percentile(H[m_cond], 5), np.percentile(H[m_cond], 95))
            ax_stack.axvspan(*h_conv_range, alpha=0.06, color=C_CONV, zorder=0)
            ax_stack.axvspan(*h_cond_range, alpha=0.06, color=C_LID, zorder=0)
            ax_stack.text(np.mean(h_conv_range), 2, 'convective\nregime',
                          fontsize=7, ha='center', color=C_CONV_DARK, alpha=0.7)
            ax_stack.text(np.mean(h_cond_range), 2, 'conductive\nregime',
                          fontsize=7, ha='center', color=C_LID_DARK, alpha=0.7)

        ax_stack.set_xlim(0, 100)
        ax_stack.set_ylim(0, 100)
        ax_stack.set_aspect('equal')
        ax_stack.legend(fontsize=9, loc='upper left', framealpha=0.9)
        ax_stack.set_ylabel('Mean layer thickness (km)', fontsize=10)
        ax_stack.set_title(f'({chr(98 + row * 2)})  {label} \u2014 stacked structure',
                           fontsize=11, fontweight='bold', loc='left')
        if row == 3:
            ax_stack.set_xlabel('Total ice shell thickness (km)', fontsize=10)

    fig.suptitle('Ice shell internal structure by ocean heat transport scenario\n'
                 r'(Ra$_{\rm crit}$ = 1000, 500 GW total budget)',
                 fontsize=14, fontweight='bold', y=0.995)

    save_path = os.path.join(FIGURES_DIR, 'shell_structure_improved.png')
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f'Saved: {save_path}')
    plt.close(fig)


def plot_regime(regime, mask_fn, title_suffix, filename, xlim_h=100, xlim_dist=90):
    """
    Plot the structure figure for a single subpopulation (convective OR conductive).

    Args:
        regime: 'convective' or 'conductive' (for labels)
        mask_fn: callable(Ra) -> bool mask selecting the subpopulation
        title_suffix: appended to the suptitle
        filename: output filename
        xlim_h: x-axis limit for thickness / stacked panels
        xlim_dist: x-axis limit for distribution panels
    """
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig, axes = plt.subplots(4, 2, figsize=(15, 18))
    fig.subplots_adjust(hspace=0.35, wspace=0.28)
    x_km = np.linspace(0, xlim_dist, 500)

    for row, (label, files, accent) in enumerate(SCENARIOS):
        data = load_and_concat(files)
        m = mask_fn(data['Ra'])
        n_total = len(data['H'])
        n_sub = m.sum()

        H = data['H'][m]
        Dc = data['Dc'][m]
        Dv = data['Dv'][m]

        ax_dist = axes[row, 0]
        ax_stack = axes[row, 1]

        if n_sub < 5:
            for ax in (ax_dist, ax_stack):
                ax.text(0.5, 0.5, f'< 5 samples\n({n_sub}/{n_total})',
                        transform=ax.transAxes, ha='center', va='center',
                        fontsize=12, color='#999')
                ax.set_title(f'{label}', fontsize=11, fontweight='bold', loc='left')
            continue

        # ── Left: layer distributions ─────────────────────────────────────
        pdf_dc = safe_kde(Dc, x_km)
        dv_pos = Dv[Dv > 0.5]
        pdf_dv = safe_kde(dv_pos, x_km)

        ax_dist.fill_between(x_km, pdf_dc, alpha=0.30, color=C_LID, zorder=2)
        ax_dist.plot(x_km, pdf_dc, color=C_LID_DARK, lw=2, zorder=3,
                     label=r'$D_{\rm cond}$ (lid)')
        ax_dist.fill_between(x_km, pdf_dv, alpha=0.30, color=C_CONV, zorder=2)
        ax_dist.plot(x_km, pdf_dv, color=C_CONV_DARK, lw=2, zorder=3,
                     label=r'$D_{\rm conv}$ (sublayer)')

        med_dc = np.median(Dc)
        med_dv = np.median(dv_pos) if len(dv_pos) > 0 else 0
        med_H = np.median(H)
        ax_dist.axvline(med_dc, color=C_LID_DARK, ls=':', lw=1.5, alpha=0.7)
        if med_dv > 0:
            ax_dist.axvline(med_dv, color=C_CONV_DARK, ls=':', lw=1.5, alpha=0.7)

        pct = 100 * n_sub / n_total
        stats = (f'N = {n_sub:,} / {n_total:,} ({pct:.0f}%)\n'
                 f'med H = {med_H:.1f} km\n'
                 f'med $D_c$ = {med_dc:.1f} km\n'
                 f'med $D_v$ = {med_dv:.1f} km\n'
                 f'med lid frac = {med_dc / med_H:.0%}')
        ax_dist.text(0.97, 0.95, stats, transform=ax_dist.transAxes,
                     fontsize=8, va='top', ha='right', fontfamily='monospace',
                     bbox=dict(boxstyle='round,pad=0.4', fc='white', alpha=0.85, ec='#ccc'))

        ax_dist.set_xlim(0, xlim_dist)
        ax_dist.set_ylim(bottom=0)
        ax_dist.legend(fontsize=9, loc='upper left', framealpha=0.9)
        ax_dist.set_ylabel('Probability density', fontsize=10)
        ax_dist.set_title(f'({chr(97 + row * 2)})  {label} \u2014 layer distributions',
                          fontsize=11, fontweight='bold', loc='left')
        if row == 3:
            ax_dist.set_xlabel('Layer thickness (km)', fontsize=10)

        # ── Right: stacked structure ──────────────────────────────────────
        centres, mean_dc, mean_dv, counts = binned_structure(H, Dc, Dv, n_bins=25)

        ax_stack.fill_between(centres, 0, mean_dc, color=C_LID, alpha=0.65,
                              label=r'$D_{\rm cond}$', zorder=2)
        ax_stack.fill_between(centres, mean_dc, mean_dc + mean_dv,
                              color=C_CONV, alpha=0.65,
                              label=r'$D_{\rm conv}$', zorder=2)
        ax_stack.plot([0, xlim_h], [0, xlim_h], 'k--', lw=1, alpha=0.3, zorder=1)
        ax_stack.plot(centres, mean_dc + mean_dv, 'k-', lw=2, alpha=0.8, zorder=4,
                      label=r'$H_{\rm total}$ (mean)')

        ax_stack.set_xlim(0, xlim_h)
        ax_stack.set_ylim(0, xlim_h)
        ax_stack.set_aspect('equal')
        ax_stack.legend(fontsize=9, loc='upper left', framealpha=0.9)
        ax_stack.set_ylabel('Mean layer thickness (km)', fontsize=10)
        ax_stack.set_title(f'({chr(98 + row * 2)})  {label} \u2014 stacked structure',
                           fontsize=11, fontweight='bold', loc='left')
        if row == 3:
            ax_stack.set_xlabel('Total ice shell thickness (km)', fontsize=10)

    fig.suptitle(f'Ice shell structure \u2014 {title_suffix}\n'
                 r'(Ra$_{\rm crit}$ = 1000, 500 GW total budget)',
                 fontsize=14, fontweight='bold', y=0.995)

    save_path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f'Saved: {save_path}')
    plt.close(fig)


if __name__ == '__main__':
    main()

    # Convective-only (Ra >= Ra_crit)
    plot_regime(
        regime='convective',
        mask_fn=lambda Ra: Ra >= RA_CRIT,
        title_suffix='CONVECTIVE branch only (Ra \u2265 Ra$_{\\rm crit}$)',
        filename='shell_structure_convective.png',
        xlim_h=80, xlim_dist=80,
    )

    # Conductive-only (Ra < Ra_crit)
    plot_regime(
        regime='conductive',
        mask_fn=lambda Ra: Ra < RA_CRIT,
        title_suffix='CONDUCTIVE branch only (Ra < Ra$_{\\rm crit}$)',
        filename='shell_structure_conductive.png',
        xlim_h=100, xlim_dist=100,
    )
