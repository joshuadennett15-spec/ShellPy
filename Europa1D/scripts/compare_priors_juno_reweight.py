"""
Two-tier prior comparison: audited 15k baseline vs wide-prior 500 sensitivity.

Reuses the importance-weight machinery from bayesian_inversion_juno.py to ask
whether the weak Juno D_cond update on the audited prior is an artefact of
prior narrowing — i.e. does the literature-envelope wide prior turn the same
Juno constraint into a meaningful filter?

Reports for each (archive, Juno center) combination:
  - Prior median + 16/84 percentiles of D_cond and H_total
  - Posterior weighted median + percentiles
  - ESS / N (Kish), and ESS fraction
  - 2-sigma prior-overlap with Juno band
  - KL-style log evidence shift (sum log-weight before normalisation)
"""
import os
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)
sys.path.insert(0, os.path.join(THIS_DIR, "..", "src"))

import numpy as np

from bayesian_inversion_juno import (
    compute_log_weights,
    normalize_weights,
    effective_sample_size,
    weighted_percentile,
)

RESULTS_DIR = os.path.join(THIS_DIR, "..", "results")

ARCHIVES = [
    ("Audited 15k (baseline)", "mc_15000_optionA_v2_andrade.npz"),
    ("Wide  500 (sensitivity)", "mc_500_wide_priors_andrade.npz"),
]

JUNO_CENTERS = [
    ("Juno best (29 km)", 29.0, 10.0),
    ("Juno alt  (24 km)", 24.0, 10.0),
]

SIGMA_MODEL = 3.0  # km — matches bayesian_inversion_juno default


def _summary_unweighted(values):
    v = values[np.isfinite(values)]
    return (
        np.percentile(v, 50),
        np.percentile(v, 15.87),
        np.percentile(v, 84.13),
    )


def _summary_weighted(values, w):
    return (
        weighted_percentile(values, w, 50),
        weighted_percentile(values, w, 15.87),
        weighted_percentile(values, w, 84.13),
    )


def _print_block(label, archive_path, D_obs, sigma_obs):
    archive = np.load(archive_path)
    D_cond = archive["D_cond_km"]
    H_total = archive["thicknesses_km"]

    finite = np.isfinite(D_cond) & np.isfinite(H_total)
    D_cond = D_cond[finite]
    H_total = H_total[finite]
    n = len(D_cond)

    log_w = compute_log_weights(D_cond, D_obs, sigma_obs, SIGMA_MODEL)
    w = normalize_weights(log_w)
    ess = effective_sample_size(w)
    log_evidence = float(np.log(np.mean(np.exp(log_w - log_w.max()))) + log_w.max())

    sigma_total = np.sqrt(sigma_obs**2 + SIGMA_MODEL**2)
    overlap_2s = float(
        np.mean(
            (D_cond > D_obs - 2 * sigma_total)
            & (D_cond < D_obs + 2 * sigma_total)
        )
    )

    pri_dc = _summary_unweighted(D_cond)
    pos_dc = _summary_weighted(D_cond, w)
    pri_h = _summary_unweighted(H_total)
    pos_h = _summary_weighted(H_total, w)

    shift_dc = pos_dc[0] - pri_dc[0]

    print(f"\n  [{label}]")
    print(f"    N valid:           {n}")
    print(f"    ESS:               {ess:.0f} / {n}  ({100 * ess / n:.1f}%)")
    print(f"    log evidence:      {log_evidence:+.2f}")
    print(f"    2-sigma overlap:   {100 * overlap_2s:.1f}%")
    print(f"    D_cond prior  med: {pri_dc[0]:5.1f} km   "
          f"[{pri_dc[1]:5.1f}, {pri_dc[2]:5.1f}]")
    print(f"    D_cond post   med: {pos_dc[0]:5.1f} km   "
          f"[{pos_dc[1]:5.1f}, {pos_dc[2]:5.1f}]   "
          f"(shift {shift_dc:+.1f} km)")
    print(f"    H_total prior med: {pri_h[0]:5.1f} km   "
          f"[{pri_h[1]:5.1f}, {pri_h[2]:5.1f}]")
    print(f"    H_total post  med: {pos_h[0]:5.1f} km   "
          f"[{pos_h[1]:5.1f}, {pos_h[2]:5.1f}]")
    return {
        "label": label,
        "n": n,
        "ess": ess,
        "log_evidence": log_evidence,
        "overlap_2s": overlap_2s,
        "pri_dc": pri_dc,
        "pos_dc": pos_dc,
        "shift_dc": shift_dc,
    }


def main():
    print("=" * 72)
    print("TWO-TIER JUNO REWEIGHT — audited baseline vs wide-prior sensitivity")
    print(f"sigma_model = {SIGMA_MODEL} km")
    print("=" * 72)

    rows = []
    for juno_label, D_obs, sigma_obs in JUNO_CENTERS:
        print(f"\n=== {juno_label}  (sigma_obs = {sigma_obs} km, "
              f"sigma_total = {np.sqrt(sigma_obs**2 + SIGMA_MODEL**2):.1f}) ===")
        for arch_label, fname in ARCHIVES:
            path = os.path.join(RESULTS_DIR, fname)
            if not os.path.exists(path):
                print(f"\n  [{arch_label}]  MISSING: {path}")
                continue
            r = _print_block(arch_label, path, D_obs, sigma_obs)
            r["juno"] = juno_label
            rows.append(r)

    print("\n" + "=" * 72)
    print("DELTA SUMMARY — does wide prior change the Juno inference?")
    print("=" * 72)
    print(f"  {'Juno':18s}  {'Prior':24s}  {'D_cond shift':>13s}  "
          f"{'ESS%':>6s}  {'log Z':>8s}")
    for r in rows:
        print(f"  {r['juno']:18s}  {r['label']:24s}  "
              f"{r['shift_dc']:+8.1f} km   "
              f"{100 * r['ess'] / r['n']:5.1f}%  "
              f"{r['log_evidence']:+8.2f}")
    print("\nDecision rule: if wide-prior shift_dc differs from audited shift_dc")
    print("by > 2 km, or wide-prior ESS% drops below ~20%, escalate to N=15k.")


if __name__ == "__main__":
    main()
