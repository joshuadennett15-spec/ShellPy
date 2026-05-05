"""
Latitude sweep of scenario Bayes factors under the Juno D_cond constraint.

For each latitude on the native 2D grid, evaluates the Juno Gaussian
likelihood L(D_cond | 29, sigma_eff) against every sample in each
scenario's 500-member MC ensemble and computes:
  * per-scenario marginal likelihood (evidence) Z(lat) = mean_i L(D_cond_i(lat))
  * pairwise Bayes factors BF_ab(lat) = Z_a(lat) / Z_b(lat)
  * effective sample size N_eff(lat) = (sum w_i)^2 / sum w_i^2

Outputs:
  * Europa2D/results/bayes_factor_latitude_sweep.npz
  * prints a latitude-marginal BF table for the six scenario pairs

Extension of Europa2D/scripts/juno_bayes_factors.py.
"""
from __future__ import annotations

import os
import sys
from itertools import combinations

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.join(_SCRIPT_DIR, "..")

RESULTS_DIR = os.path.join(_PROJECT_DIR, "results")
OUTPUT_NPZ = os.path.join(RESULTS_DIR, "bayes_factor_latitude_sweep.npz")

N_ITER = 500

# Juno MWR D_cond constraint (Levin et al. 2025, pure water ice).
JUNO_DCOND_KM = 29.0
JUNO_DCOND_SIGMA_OBS = 10.0
MODEL_DISCREPANCY = 3.0
SIGMA_EFF = float(np.sqrt(JUNO_DCOND_SIGMA_OBS**2 + MODEL_DISCREPANCY**2))

SCENARIOS = [
    ("uniform_transport", "Uniform"),
    ("soderlund2014_equator", "Equator-enhanced"),
    ("lemasquerier2023_polar", "Polar-enhanced"),
    ("lemasquerier2023_polar_strong", "Strong polar"),
]


def _load(key: str) -> dict:
    path = os.path.join(RESULTS_DIR, f"mc_2d_{key}_{N_ITER}.npz")
    return dict(np.load(path, allow_pickle=True))


def _gaussian_likelihood(d_cond: np.ndarray, mu: float = JUNO_DCOND_KM,
                        sigma: float = SIGMA_EFF) -> np.ndarray:
    return np.exp(-0.5 * ((d_cond - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


def sweep() -> dict:
    data = {key: _load(key) for key, _ in SCENARIOS}
    lats = data[SCENARIOS[0][0]]["latitudes_deg"]
    n_lat = len(lats)

    evidence = np.zeros((len(SCENARIOS), n_lat))
    n_eff = np.zeros((len(SCENARIOS), n_lat))
    dc_prior_median = np.zeros((len(SCENARIOS), n_lat))
    dc_post_median = np.zeros((len(SCENARIOS), n_lat))

    for i, (key, _) in enumerate(SCENARIOS):
        Dc = data[key]["D_cond_profiles"]   # (N, n_lat)
        for j in range(n_lat):
            dc_j = Dc[:, j]
            lk = _gaussian_likelihood(dc_j)
            evidence[i, j] = float(np.mean(lk))
            w_sum = lk.sum()
            if w_sum > 0:
                w = lk / w_sum
                n_eff[i, j] = 1.0 / float(np.sum(w ** 2))
                # weighted median via sort
                order = np.argsort(dc_j)
                cdf = np.cumsum(w[order])
                dc_post_median[i, j] = float(dc_j[order][np.searchsorted(cdf, 0.5)])
            else:
                n_eff[i, j] = 0.0
                dc_post_median[i, j] = float(np.median(dc_j))
            dc_prior_median[i, j] = float(np.median(dc_j))

    pairs = list(combinations(range(len(SCENARIOS)), 2))
    bf = np.zeros((len(pairs), n_lat))
    for p, (a, b) in enumerate(pairs):
        with np.errstate(divide="ignore", invalid="ignore"):
            bf[p] = np.where(evidence[b] > 0, evidence[a] / evidence[b], np.inf)

    return {
        "latitudes_deg": lats,
        "scenario_keys": np.array([k for k, _ in SCENARIOS]),
        "scenario_labels": np.array([lbl for _, lbl in SCENARIOS]),
        "pair_indices": np.array(pairs),
        "evidence": evidence,
        "bayes_factor": bf,
        "n_eff": n_eff,
        "n_samples": N_ITER,
        "dc_prior_median": dc_prior_median,
        "dc_post_median": dc_post_median,
        "juno_mu_km": JUNO_DCOND_KM,
        "juno_sigma_eff_km": SIGMA_EFF,
    }


def print_summary(out: dict) -> None:
    lats = out["latitudes_deg"]
    labels = out["scenario_labels"]
    pairs = out["pair_indices"]
    bf = out["bayes_factor"]

    print("=" * 78)
    print(f"Juno D_cond = {JUNO_DCOND_KM} +/- {SIGMA_EFF:.2f} km "
          f"(sigma_obs = {JUNO_DCOND_SIGMA_OBS}, model_disc = {MODEL_DISCREPANCY})")
    print(f"N_samples per scenario = {N_ITER}")
    print("=" * 78)

    key_lats = [0.0, 17.5, 35.0, 52.5, 70.0, 87.5]
    idx_lats = [int(np.argmin(np.abs(lats - l))) for l in key_lats]

    print("\nBayes factors (row/col) at 35 deg latitude:")
    idx35 = int(np.argmin(np.abs(lats - 35.0)))
    mat = np.ones((len(labels), len(labels)))
    for p, (a, b) in enumerate(pairs):
        mat[a, b] = bf[p, idx35]
        mat[b, a] = 1.0 / bf[p, idx35] if bf[p, idx35] > 0 else np.inf
    hdr = " " * 22 + "".join(f"{lbl[:14]:>15s}" for lbl in labels)
    print(hdr)
    for i, lbl in enumerate(labels):
        row = f"{lbl[:20]:>20s}  " + "".join(f"{mat[i, j]:>15.3f}" for j in range(len(labels)))
        print(row)

    print("\nBF and N_eff vs latitude (most-separated pair Uniform vs Strong polar):")
    pair_key = (0, 3)
    p_idx = [i for i, pr in enumerate(pairs) if tuple(pr) == pair_key][0]
    print(f"{'lat (deg)':>10}  {'BF (U/SP)':>10}  {'log10 BF':>10}  "
          f"{'N_eff/N U':>10}  {'N_eff/N SP':>12}")
    for idx in idx_lats:
        lat = lats[idx]
        val = bf[p_idx, idx]
        lg = np.log10(val) if val > 0 else -np.inf
        ne_u = out["n_eff"][0, idx] / N_ITER
        ne_s = out["n_eff"][3, idx] / N_ITER
        print(f"{lat:>10.1f}  {val:>10.3f}  {lg:>10.3f}  {ne_u:>10.3f}  {ne_s:>12.3f}")

    # threshold crossings for each pair
    print("\nLowest |lat| at which any pair crosses BF = 3 (substantial evidence):")
    for p, (a, b) in enumerate(pairs):
        excess = np.maximum(bf[p], 1.0 / np.where(bf[p] > 0, bf[p], np.inf))
        over = np.where(excess >= 3.0)[0]
        if over.size:
            lat_min = float(np.min(np.abs(lats[over])))
            print(f"  {labels[a]:<20s} vs {labels[b]:<20s}  : "
                  f"|lat| = {lat_min:.1f} deg  (max |log10 BF| = {np.max(np.abs(np.log10(bf[p]))):.2f})")
        else:
            print(f"  {labels[a]:<20s} vs {labels[b]:<20s}  : never crosses BF = 3  "
                  f"(max |log10 BF| = {np.max(np.abs(np.log10(bf[p]))):.3f})")


def main() -> None:
    out = sweep()
    os.makedirs(os.path.dirname(OUTPUT_NPZ), exist_ok=True)
    np.savez(OUTPUT_NPZ, **out)
    print(f"Wrote {OUTPUT_NPZ}")
    print_summary(out)


if __name__ == "__main__":
    main()
