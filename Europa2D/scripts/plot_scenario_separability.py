"""
Scenario separability analysis for thesis Chapter 4, Section 4.8.

Produces two figures:
  1. fig_scenario_separation.pdf  — Pairwise |Δ median| for D_cond and H_total vs latitude
  2. fig_precision_threshold.pdf  — Minimum σ_eff for BF > 3 vs latitude

Data source: existing 500-sample 2D MC archives (no new runs required).
"""
import os
import sys
from itertools import combinations

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logsumexp

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.join(_SCRIPT_DIR, "..")
sys.path.insert(0, os.path.join(_PROJECT_DIR, "src"))
sys.path.insert(0, os.path.join(_PROJECT_DIR, "..", "Europa1D", "src"))
sys.path.insert(0, _SCRIPT_DIR)

from pub_style import apply_style, PAL, save_fig, DOUBLE_COL, SINGLE_COL

RESULTS_DIR = os.path.join(_PROJECT_DIR, "results")
FIGURES_DIR = os.path.join(_PROJECT_DIR, "figures", "thesis")
N_ITER = 500

JUNO_SIGMA = 10.4  # km — current Juno MWR effective uncertainty
JUNO_LAT = 35.0

SCENARIOS = [
    ("uniform_transport",             "Uniform"),
    ("soderlund2014_equator",         "Eq-enhanced"),
    ("lemasquerier2023_polar",        "Polar-enh."),
    ("lemasquerier2023_polar_strong", "Strong polar"),
]

# Colours for scenario pairs — use a qualitative palette
PAIR_COLOURS = [
    "#1b9e77", "#d95f02", "#7570b3",
    "#e7298a", "#66a61e", "#e6ab02",
]


def _load(scenario_key):
    path = os.path.join(RESULTS_DIR, f"mc_2d_{scenario_key}_{N_ITER}.npz")
    return dict(np.load(path, allow_pickle=True))


def _interp_at(lat_arr, profiles, target_lat):
    """Interpolate per-sample profiles at a single latitude."""
    return np.array([
        np.interp(target_lat, lat_arr, profiles[i])
        for i in range(profiles.shape[0])
    ])


def _log_marginal_likelihood(samples, obs, sigma):
    """
    Log marginal likelihood for a Gaussian observation applied to an ensemble.

    log p(obs | model) ≈ log(1/N Σ_i N(obs; sample_i, σ²))
                       = logsumexp(-0.5 * ((obs - samples)/σ)²) - log(N) - log(σ√2π)
    """
    n = len(samples)
    z = (obs - samples) / sigma
    return logsumexp(-0.5 * z**2) - np.log(n) - 0.5 * np.log(2 * np.pi) - np.log(sigma)


def compute_separation(data_dict, quantity_key):
    """
    Compute pairwise |Δ median| for a given quantity at each latitude.

    Returns: latitudes, dict of (pair_label -> separation_array)
    """
    lat = data_dict[SCENARIOS[0][0]]["latitudes_deg"]
    medians = {}
    for key, label in SCENARIOS:
        profiles = data_dict[key][quantity_key]
        medians[key] = np.median(profiles, axis=0)

    separations = {}
    for (k1, l1), (k2, l2) in combinations(SCENARIOS, 2):
        pair_label = f"{l1} vs {l2}"
        separations[pair_label] = np.abs(medians[k1] - medians[k2])

    return lat, separations


def compute_precision_thresholds(data_dict, quantity_key, bf_target=3.0):
    """
    At each latitude, find the minimum σ_eff for which the most-separated
    scenario pair yields BF > bf_target.

    Uses bisection on σ_eff in [0.1, 50] km.
    """
    lat = data_dict[SCENARIOS[0][0]]["latitudes_deg"]
    n_lat = len(lat)
    thresholds = np.full(n_lat, np.nan)

    for j in range(n_lat):
        # Extract samples at this latitude for each scenario
        samples_at_lat = {}
        for key, _ in SCENARIOS:
            profiles = data_dict[key][quantity_key]
            samples_at_lat[key] = profiles[:, j]

        # Find the pair with maximum median separation at this latitude
        max_bf_at_sigma = 0.0
        best_pair = None
        for (k1, _), (k2, _) in combinations(SCENARIOS, 2):
            med_diff = abs(np.median(samples_at_lat[k1]) - np.median(samples_at_lat[k2]))
            if med_diff > max_bf_at_sigma:
                max_bf_at_sigma = med_diff
                best_pair = (k1, k2)

        if best_pair is None:
            continue

        k1, k2 = best_pair
        s1 = samples_at_lat[k1]
        s2 = samples_at_lat[k2]

        # Use midpoint of the two medians as the hypothetical observation
        obs = 0.5 * (np.median(s1) + np.median(s2))

        # Bisection: find σ where BF = bf_target
        sigma_lo, sigma_hi = 0.1, 50.0

        # Check if BF > target is achievable at sigma_lo
        log_ml1 = _log_marginal_likelihood(s1, obs, sigma_lo)
        log_ml2 = _log_marginal_likelihood(s2, obs, sigma_lo)
        bf_lo = np.exp(abs(log_ml1 - log_ml2))
        if bf_lo < bf_target:
            # Even at maximum precision, can't discriminate
            thresholds[j] = np.nan
            continue

        # Check BF at sigma_hi
        log_ml1 = _log_marginal_likelihood(s1, obs, sigma_hi)
        log_ml2 = _log_marginal_likelihood(s2, obs, sigma_hi)
        bf_hi = np.exp(abs(log_ml1 - log_ml2))
        if bf_hi >= bf_target:
            # Even at coarsest precision, can discriminate
            thresholds[j] = sigma_hi
            continue

        # Bisect
        for _ in range(50):
            sigma_mid = 0.5 * (sigma_lo + sigma_hi)
            log_ml1 = _log_marginal_likelihood(s1, obs, sigma_mid)
            log_ml2 = _log_marginal_likelihood(s2, obs, sigma_mid)
            bf_mid = np.exp(abs(log_ml1 - log_ml2))
            if bf_mid > bf_target:
                sigma_lo = sigma_mid
            else:
                sigma_hi = sigma_mid
            if sigma_hi - sigma_lo < 0.01:
                break

        thresholds[j] = 0.5 * (sigma_lo + sigma_hi)

    return lat, thresholds


def plot_scenario_separation(data_dict):
    """Figure 4.13: pairwise H_total scenario separation vs latitude."""
    apply_style()
    fig, ax = plt.subplots(figsize=(SINGLE_COL, 0.75 * SINGLE_COL))

    lat, seps = compute_separation(data_dict, "H_profiles")
    for i, (pair_label, sep_arr) in enumerate(seps.items()):
        ax.plot(lat, sep_arr, color=PAIR_COLOURS[i % len(PAIR_COLOURS)],
                linewidth=1.2, label=pair_label)
    ax.set_xlabel(r"Latitude ($^\circ$)")
    ax.set_ylabel(r"$|\Delta\,\mathrm{median}\;H_\mathrm{total}|$ (km)")
    ax.set_xlim(0, 90)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=6, loc="upper right", framealpha=0.9)

    fig.tight_layout()
    save_fig(fig, "fig_scenario_separation", FIGURES_DIR)
    print("  -> fig_scenario_separation.pdf")
    plt.close(fig)


def plot_precision_threshold(data_dict):
    """Figure 13: Minimum σ_eff for BF > 3 vs latitude."""
    apply_style()
    fig, ax = plt.subplots(figsize=(SINGLE_COL, 0.75 * SINGLE_COL))

    for qty, colour, label in [
        ("D_cond_profiles", PAL.BLUE, r"$D_\mathrm{cond}$-based"),
        ("H_profiles", PAL.RED, r"$H_\mathrm{total}$-based"),
    ]:
        lat, thresholds = compute_precision_thresholds(data_dict, qty)
        valid = ~np.isnan(thresholds)
        ax.plot(lat[valid], thresholds[valid], color=colour, linewidth=1.5,
                label=label)
        # Mark unachievable latitudes
        if np.any(~valid):
            for k in np.where(~valid)[0]:
                ax.plot(lat[k], 0.5, marker="x", color=colour, markersize=4,
                        alpha=0.5)

    ax.axhline(JUNO_SIGMA, color="grey", linestyle="--", linewidth=0.8,
               label=rf"Juno $\sigma_{{\mathrm{{eff}}}}$ = {JUNO_SIGMA} km")

    ax.set_xlabel(r"Latitude ($^\circ$)")
    ax.set_ylabel(r"$\sigma_\mathrm{eff}$ for BF $> 3$ (km)")
    ax.set_xlim(0, 90)
    ax.set_ylim(0, 20)
    ax.legend(fontsize=7, loc="upper left", framealpha=0.9)

    fig.tight_layout()
    save_fig(fig, "fig_precision_threshold", FIGURES_DIR)
    print("  -> fig_precision_threshold.pdf")
    plt.close(fig)


def main():
    print("Loading 2D MC archives...")
    data_dict = {}
    for key, label in SCENARIOS:
        data_dict[key] = _load(key)
        n = data_dict[key]["H_profiles"].shape[0]
        print(f"  {label}: {n} samples, "
              f"{len(data_dict[key]['latitudes_deg'])} latitude nodes")

    print("\nGenerating scenario separation figure...")
    plot_scenario_separation(data_dict)

    print("Generating precision threshold figure...")
    plot_precision_threshold(data_dict)

    print("\nDone.")


if __name__ == "__main__":
    main()
