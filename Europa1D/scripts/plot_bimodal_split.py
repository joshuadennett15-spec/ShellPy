"""
Plot bimodal subpopulation split for each ocean heat transport scenario.

Splits MC results into convective (Ra >= Ra_crit) and conductive (Ra < Ra_crit)
subpopulations following the Mitri & Showman (2005) bistability framework.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
RA_CRIT = 1000.0

# ── Scenario definitions ─────────────────────────────────────────────────────
# Each scenario: (label, list of .npz files to concatenate, color)
SCENARIOS = [
    ('Global',              ['budget_global'],                                        '#333333'),
    ('Uniform',             ['budget_uniform_equator', 'budget_uniform_pole'],        '#1B9E77'),
    ('Equatorial-enhanced', ['budget_soderlund2014_equator', 'budget_soderlund2014_pole'], '#D95F02'),
    ('Polar-enhanced',      ['budget_lemasquerier2023_equator', 'budget_lemasquerier2023_pole'], '#377EB8'),
]


def load_and_concat(file_list):
    """Load and concatenate multiple .npz result files."""
    all_H, all_Ra, all_Dcond, all_Dconv, all_lid = [], [], [], [], []
    for fname in file_list:
        path = os.path.join(RESULTS_DIR, f'{fname}.npz')
        d = np.load(path)
        all_H.append(d['thicknesses_km'])
        all_Ra.append(d['Ra_values'])
        all_Dcond.append(d['D_cond_km'])
        all_Dconv.append(d['D_conv_km'])
        all_lid.append(d['lid_fractions'])
    return {
        'H': np.concatenate(all_H),
        'Ra': np.concatenate(all_Ra),
        'D_cond': np.concatenate(all_Dcond),
        'D_conv': np.concatenate(all_Dconv),
        'lid': np.concatenate(all_lid),
    }


def kde_smooth(data, x_grid):
    """KDE with fallback for tiny or degenerate samples."""
    if len(data) < 3:
        return np.zeros_like(x_grid)
    # Guard against zero-variance data (e.g., all lid fractions = 1.0)
    if np.std(data) < 1e-10:
        return np.zeros_like(x_grid)
    try:
        kde = gaussian_kde(data)
        return kde(x_grid)
    except np.linalg.LinAlgError:
        return np.zeros_like(x_grid)


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # ── Load all scenarios ────────────────────────────────────────────────────
    scenarios = []
    for label, files, color in SCENARIOS:
        data = load_and_concat(files)
        mask_conv = data['Ra'] >= RA_CRIT
        mask_cond = ~mask_conv
        scenarios.append((label, data, mask_conv, mask_cond, color))

    # ── Figure: 4×2 grid (one row per scenario, left=thickness, right=lid fraction)
    fig, axes = plt.subplots(4, 2, figsize=(14, 16), constrained_layout=True)

    x_H = np.linspace(0, 100, 400)
    x_lid = np.linspace(0, 1, 400)

    for row, (label, data, m_conv, m_cond, color) in enumerate(scenarios):
        ax_h = axes[row, 0]
        ax_lid = axes[row, 1]

        H = data['H']
        lid = data['lid']
        n_conv = m_conv.sum()
        n_cond = m_cond.sum()
        n_total = len(H)
        f_conv = n_conv / n_total
        f_cond = n_cond / n_total

        # ── Left panel: thickness PDF ─────────────────────────────────────────
        # Combined
        pdf_all = kde_smooth(H, x_H)
        ax_h.fill_between(x_H, pdf_all, alpha=0.12, color=color)
        ax_h.plot(x_H, pdf_all, color=color, lw=1.5, ls='--', alpha=0.5,
                  label=f'Combined (N={n_total})')

        # Convective subpopulation (weighted by fraction)
        if n_conv > 2:
            pdf_conv = kde_smooth(H[m_conv], x_H) * f_conv
            ax_h.fill_between(x_H, pdf_conv, alpha=0.35, color='#E74C3C')
            ax_h.plot(x_H, pdf_conv, color='#C0392B', lw=2,
                      label=f'Convective ({n_conv}, {f_conv:.0%}) '
                            f'med={np.median(H[m_conv]):.0f} km')

        # Conductive subpopulation (weighted by fraction)
        if n_cond > 2:
            pdf_cond = kde_smooth(H[m_cond], x_H) * f_cond
            ax_h.fill_between(x_H, pdf_cond, alpha=0.35, color='#3498DB')
            ax_h.plot(x_H, pdf_cond, color='#2471A3', lw=2,
                      label=f'Conductive ({n_cond}, {f_cond:.0%}) '
                            f'med={np.median(H[m_cond]):.0f} km')

        ax_h.set_xlim(0, 100)
        ax_h.set_ylim(bottom=0)
        ax_h.set_ylabel('Probability density')
        ax_h.legend(fontsize=8, loc='upper right')
        ax_h.set_title(f'{label}', fontsize=12, fontweight='bold', loc='left')
        if row == 3:
            ax_h.set_xlabel('Total ice shell thickness (km)')

        # ── Right panel: lid fraction PDF ─────────────────────────────────────
        pdf_lid_all = kde_smooth(lid, x_lid)
        ax_lid.fill_between(x_lid, pdf_lid_all, alpha=0.12, color=color)
        ax_lid.plot(x_lid, pdf_lid_all, color=color, lw=1.5, ls='--', alpha=0.5,
                    label=f'Combined (med={np.median(lid):.0%})')

        if n_conv > 2:
            pdf_lid_conv = kde_smooth(lid[m_conv], x_lid) * f_conv
            ax_lid.fill_between(x_lid, pdf_lid_conv, alpha=0.35, color='#E74C3C')
            ax_lid.plot(x_lid, pdf_lid_conv, color='#C0392B', lw=2,
                        label=f'Convective (med={np.median(lid[m_conv]):.0%})')

        if n_cond > 2:
            pdf_lid_cond = kde_smooth(lid[m_cond], x_lid) * f_cond
            ax_lid.fill_between(x_lid, pdf_lid_cond, alpha=0.35, color='#3498DB')
            ax_lid.plot(x_lid, pdf_lid_cond, color='#2471A3', lw=2,
                        label=f'Conductive (med={np.median(lid[m_cond]):.0%})')

        ax_lid.set_xlim(0, 1)
        ax_lid.set_ylim(bottom=0)
        ax_lid.set_ylabel('Probability density')
        ax_lid.legend(fontsize=8, loc='upper left')
        if row == 3:
            ax_lid.set_xlabel(r'Lid fraction ($D_{\rm cond} / H_{\rm total}$)')

    fig.suptitle(
        r'Convective vs conductive subpopulations (Ra$_{\rm crit}$ = '
        f'{RA_CRIT:.0f}, 500 GW budget)',
        fontsize=14, fontweight='bold', y=1.01,
    )

    save_path = os.path.join(FIGURES_DIR, 'bimodal_split.png')
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f'Saved: {save_path}')
    plt.close(fig)


if __name__ == '__main__':
    main()
