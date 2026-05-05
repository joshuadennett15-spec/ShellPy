"""
Bayesian Juno comparison across equatorial-proxy modes.

For each mode, runs importance reweighting against D_cond for both
Juno observation models (pure-water 29+/-10, low-salinity 24+/-10).
Computes marginal likelihoods and reports Bayes factors between modes.

Reuses core machinery from bayesian_inversion_juno.py.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS_DIR)

import numpy as np
from scipy.special import logsumexp
import matplotlib.pyplot as plt

from bayesian_inversion_juno import (
    compute_log_weights, normalize_weights, effective_sample_size,
    weighted_percentile, posterior_summary, run_model,
)
from pub_style import (
    apply_style, PAL, figsize_double, save_fig, add_minor_gridlines,
    label_panel,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures', 'pub')

apply_style()

MODES = [
    ("eq_depleted_strong_andrade.npz", "Depleted strong (0.55x)", "depleted_strong"),
    ("eq_depleted_andrade.npz",        "Depleted (0.67x)",        "depleted"),
    ("eq_baseline_andrade.npz",        "Baseline (1.0x)",         "baseline"),
    ("eq_moderate_andrade.npz",        "Moderate (1.2x)",         "moderate"),
    ("eq_strong_andrade.npz",          "Strong (1.5x)",           "strong"),
]

JUNO_OBS = [
    (29.0, 10.0, "Model A: 29\u00b110 km"),
    (24.0, 10.0, "Model B: 24\u00b110 km"),
]

SIGMA_MODEL = 3.0


def log_marginal_likelihood(D_cond, D_obs, sigma_obs, sigma_model):
    """
    Log marginal likelihood: log p(D_obs | mode) = log (1/N) sum_i p(D_obs | D_cond_i).

    Uses logsumexp for numerical stability.
    """
    log_w = compute_log_weights(D_cond, D_obs, sigma_obs, sigma_model)
    return logsumexp(log_w) - np.log(len(log_w))


def main():
    np.random.seed(42)

    results_table = []

    for filename, mode_label, tag in MODES:
        filepath = os.path.join(RESULTS_DIR, filename)
        if not os.path.exists(filepath):
            print(f"Skipping {mode_label}: {filepath} not found")
            continue

        print(f"\n{'#' * 60}")
        print(f"# MODE: {mode_label}")
        print(f"{'#' * 60}")

        data = np.load(filepath)
        D_cond = data['D_cond_km']
        n_valid = len(D_cond)
        n_total = int(data['n_iterations']) if 'n_iterations' in data else n_valid
        print(f"  N = {n_valid:,} valid / {n_total:,} total ({100*n_valid/n_total:.0f}%)")

        row = {'mode': mode_label, 'tag': tag, 'n': n_valid, 'n_total': n_total}

        for D_obs, sigma_obs, obs_label in JUNO_OBS:
            obs_tag = "A" if D_obs == 29.0 else "B"

            log_ml = log_marginal_likelihood(D_cond, D_obs, sigma_obs, SIGMA_MODEL)
            row[f'log_ml_{obs_tag}'] = log_ml

            log_w = compute_log_weights(D_cond, D_obs, sigma_obs, SIGMA_MODEL)
            w = normalize_weights(log_w)
            ess = effective_sample_size(w)
            row[f'ess_{obs_tag}'] = ess

            print(f"\n  {obs_label}:")
            print(f"    Log marginal likelihood: {log_ml:.3f}")
            print(f"    ESS: {ess:.0f} / {n_valid} ({100 * ess / n_valid:.1f}%)")

            med_dc = weighted_percentile(D_cond, w, 50)
            row[f'D_cond_median_{obs_tag}'] = med_dc
            print(f"    Posterior D_cond median: {med_dc:.1f} km")

            if 'Ra_values' in data:
                conv_frac = np.sum(w[data['Ra_values'] >= 1000])
                row[f'conv_frac_{obs_tag}'] = conv_frac
                print(f"    Posterior convective fraction: {conv_frac:.1%}")

            # Per-mode Bayesian figures (prefixed to avoid collision)
            full_label = f"eq_{tag}_{obs_tag}"
            run_model(data, D_obs, sigma_obs, SIGMA_MODEL, full_label)

        results_table.append(row)

    # Bayes factor table
    # NOTE: These Bayes factors are conditional on solver-valid draws.
    # If valid yield differs across modes, the comparison reflects
    # p(Juno | mode, valid) not full p(Juno | mode). The yield
    # difference is reported separately as additional context.
    if len(results_table) < 2:
        print("\nNot enough modes for Bayes factor comparison.")
        return

    print(f"\n{'=' * 60}")
    print("BAYES FACTOR TABLE (relative to baseline, conditional on valid draws)")
    print(f"{'=' * 60}")

    baseline = next(r for r in results_table if r['tag'] == 'baseline')
    header = (f"{'Mode':<22s} | {'N_valid':>7s} | {'Yield':>6s} | "
              f"{'log BF_A':>10s} | {'BF_A':>8s} | {'log BF_B':>10s} | {'BF_B':>8s}")
    print(header)
    print("-" * len(header))

    for row in results_table:
        for obs_tag in ["A", "B"]:
            key = f'log_ml_{obs_tag}'
            row[f'log_bf_{obs_tag}'] = row[key] - baseline[key]

        log_bf_a = row['log_bf_A']
        log_bf_b = row['log_bf_B']
        bf_a = np.exp(log_bf_a)
        bf_b = np.exp(log_bf_b)
        n_valid = row['n']
        n_total = row.get('n_total', n_valid)
        yield_pct = f"{100 * n_valid / n_total:.0f}%" if n_total > 0 else "?"
        print(f"{row['mode']:<22s} | {n_valid:>7d} | {yield_pct:>6s} | "
              f"{log_bf_a:>10.3f} | {bf_a:>8.3f} | {log_bf_b:>10.3f} | {bf_b:>8.3f}")

    # Summary figure: BF bar chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize_double(0.40))

    labels = [r['mode'] for r in results_table]
    x = np.arange(len(labels))

    for ax, obs_tag, obs_label in [(ax1, "A", "Pure water (29 km)"),
                                    (ax2, "B", "Low salinity (24 km)")]:
        log_bfs = [r[f'log_bf_{obs_tag}'] for r in results_table]
        colors = [PAL.GREEN if bf > 0 else PAL.RED for bf in log_bfs]
        ax.bar(x, log_bfs, color=colors, alpha=0.7, edgecolor="0.3", lw=0.5)
        ax.axhline(0, color="0.5", lw=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=6.5, rotation=15, ha="right")
        ax.set_ylabel("log Bayes factor vs baseline")
        ax.set_title(obs_label, fontsize=8)
        add_minor_gridlines(ax, axis="y")

    label_panel(ax1, "a")
    label_panel(ax2, "b")

    fig.suptitle("Equatorial mode evidence comparison", fontsize=9, y=1.02)
    fig.tight_layout(w_pad=2.0)
    save_fig(fig, "fig_eq_bayes_factors", FIGURES_DIR)

    print(f"\nAll equatorial Juno figures saved to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
