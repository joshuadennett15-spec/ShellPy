#!/usr/bin/env python
"""
Interval plot: Juno refit prior vs posterior at 35 deg latitude.

Shows actual parameter ranges (prior clip range and posterior 90% CI)
so the reader can judge how much real constraint the Juno data provides.

Poster-sized: single wide row with both parameters side by side.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pub_style import apply_style, PAL, DOUBLE_COL, save_fig

apply_style()

# Scale everything up for poster legibility
SCALE = 1.8
plt.rcParams.update({
    "font.size":         8 * SCALE,
    "axes.labelsize":    9 * SCALE,
    "axes.titlesize":    9 * SCALE,
    "xtick.labelsize":   7.5 * SCALE,
    "ytick.labelsize":   7.5 * SCALE,
    "legend.fontsize":   7 * SCALE,
    "axes.linewidth":    0.6 * SCALE,
    "xtick.major.width": 0.5 * SCALE,
    "ytick.major.width": 0.5 * SCALE,
    "xtick.major.size":  3 * SCALE,
    "ytick.major.size":  3 * SCALE,
    "lines.linewidth":   1.2 * SCALE,
    "patch.linewidth":   0.5 * SCALE,
})

# ── Constants ──
BASE = os.path.join(os.path.dirname(__file__), "..", "results", "midlat_juno")
A_EUROPA = 3.09e13  # m^2
SCENARIOS = ["uniform", "soderlund", "lemasquerier"]

# Prior bounds (from INITIAL_PRIORS in run_midlat_juno_refit.py)
PRIOR_Q_LO, PRIOR_Q_HI = 5.0, 25.0       # mW/m^2
PRIOR_DG_LO, PRIOR_DG_HI = 0.05, 3.0     # mm
PRIOR_DG_CENTER = 0.60                     # mm (log-center)


# ── Helpers ──
def compute_weights(D_cond_km, D_obs_km=20.0, sigma_obs_km=4.0, sigma_model_km=2.0):
    sigma_total = np.sqrt(sigma_obs_km**2 + sigma_model_km**2)
    log_w = -0.5 * ((D_cond_km - D_obs_km) / sigma_total) ** 2
    log_w -= np.max(log_w)
    w = np.exp(log_w)
    return w / w.sum()


def weighted_percentile(vals, weights, q):
    idx = np.argsort(vals)
    sv, sw = vals[idx], weights[idx]
    cw = np.cumsum(sw)
    cw /= cw[-1]
    return np.interp(q / 100.0, cw, sv)


def extract_q_basal(data):
    q_tidal = data["param_P_tidal"] / A_EUROPA
    q_rad_vol = data["param_H_rad"] * data["thicknesses_km"] * 1e3 * 917
    return q_tidal + q_rad_vol


# ── Collect posterior stats (average across 3 nearly-identical scenarios) ──
q_meds, q_05s, q_95s, q_16s, q_84s = [], [], [], [], []
dg_meds, dg_05s, dg_95s, dg_16s, dg_84s = [], [], [], [], []

for scen in SCENARIOS:
    rc = np.load(os.path.join(BASE, f"midlat35_{scen}_constrained.npz"))
    wc = compute_weights(rc["D_cond_km"])
    qc = extract_q_basal(rc) * 1e3   # -> mW/m^2
    dgc = rc["param_d_grain"] * 1e3   # -> mm

    for arr, stores in [
        (qc, (q_meds, q_05s, q_16s, q_84s, q_95s)),
        (dgc, (dg_meds, dg_05s, dg_16s, dg_84s, dg_95s)),
    ]:
        for pct, store in zip([50, 5, 16, 84, 95], stores):
            store.append(weighted_percentile(arr, wc, pct))

q_med = np.mean(q_meds)
q_05, q_95 = np.mean(q_05s), np.mean(q_95s)
q_16, q_84 = np.mean(q_16s), np.mean(q_84s)
dg_med = np.mean(dg_meds)
dg_05, dg_95 = np.mean(dg_05s), np.mean(dg_95s)
dg_16, dg_84 = np.mean(dg_16s), np.mean(dg_84s)


# ── Plot: one row, two columns ──
fig, (ax_q, ax_dg) = plt.subplots(
    1, 2, figsize=(DOUBLE_COL * SCALE, 2.2 * SCALE),
    gridspec_kw={"wspace": 0.45},
)

y_prior = 1.0
y_post = 0.0
yticks = [y_post, y_prior]
ylabels = ["Posterior", "Prior"]
bar_h = 0.38
lw = 1.0 * SCALE
med_ms = 14 * SCALE
center_ms = 14 * SCALE
fs_annot = 7 * SCALE
fs_ci = 6.5 * SCALE


# ── q_basal panel ──
# Prior
ax_q.barh(y_prior, PRIOR_Q_HI - PRIOR_Q_LO, bar_h,
          left=PRIOR_Q_LO, color=PAL.BLUE, alpha=0.18,
          edgecolor=PAL.BLUE, linewidth=lw)
q_prior_center = (PRIOR_Q_LO + PRIOR_Q_HI) / 2
ax_q.plot(q_prior_center, y_prior, "|", color=PAL.BLUE,
          ms=center_ms, mew=lw * 1.2)

# Posterior 90% CI
ax_q.barh(y_post, q_95 - q_05, bar_h,
          left=q_05, color=PAL.BLUE, alpha=0.65,
          edgecolor=PAL.BLUE, linewidth=lw)
# Posterior 68% CI
ax_q.barh(y_post, q_84 - q_16, bar_h * 0.55,
          left=q_16, color=PAL.BLUE, alpha=0.92,
          edgecolor="none")
# Median
ax_q.plot(q_med, y_post, "|", color="white", ms=med_ms, mew=lw * 1.5)

ax_q.set_yticks(yticks)
ax_q.set_yticklabels(ylabels)
ax_q.set_ylim(-0.8, 1.8)
ax_q.set_xlabel("$q_{\\mathrm{basal}}$  (mW m$^{-2}$)")
ax_q.set_xlim(0, 28)
ax_q.set_title("$q_{\\mathrm{basal}}$: modest constraint", loc="left")

# Annotations
ax_q.text(q_med, y_post - 0.42, f"median {q_med:.1f}", ha="center",
          fontsize=fs_annot, color=PAL.BLUE, fontweight="bold")
ax_q.text((q_05 + q_95) / 2, y_post + 0.40,
          f"90% CI: [{q_05:.1f}, {q_95:.1f}]",
          ha="center", fontsize=fs_ci, color="0.35")
ax_q.text((PRIOR_Q_LO + PRIOR_Q_HI) / 2, y_prior + 0.40,
          f"[{PRIOR_Q_LO:.0f}, {PRIOR_Q_HI:.0f}]",
          ha="center", fontsize=fs_ci, color="0.35")


# ── d_grain panel (log scale) ──
ax_dg.set_xscale("log")

# Prior
ax_dg.barh(y_prior, PRIOR_DG_HI - PRIOR_DG_LO, bar_h,
           left=PRIOR_DG_LO, color=PAL.ORANGE, alpha=0.18,
           edgecolor=PAL.ORANGE, linewidth=lw)
ax_dg.plot(PRIOR_DG_CENTER, y_prior, "|", color=PAL.ORANGE,
           ms=center_ms, mew=lw * 1.2)

# Posterior 90% CI
ax_dg.barh(y_post, dg_95 - dg_05, bar_h,
           left=dg_05, color=PAL.ORANGE, alpha=0.65,
           edgecolor=PAL.ORANGE, linewidth=lw)
# Posterior 68% CI
ax_dg.barh(y_post, dg_84 - dg_16, bar_h * 0.55,
           left=dg_16, color=PAL.ORANGE, alpha=0.92,
           edgecolor="none")
# Median
ax_dg.plot(dg_med, y_post, "|", color="white", ms=med_ms, mew=lw * 1.5)

ax_dg.set_yticks(yticks)
ax_dg.set_yticklabels(ylabels)
ax_dg.set_ylim(-0.8, 1.8)
ax_dg.set_xlabel("$d_{\\mathrm{grain}}$  (mm)")
ax_dg.set_xlim(0.03, 5)
ax_dg.set_title("$d_{\\mathrm{grain}}$: tails trimmed, core still broad",
                loc="left")

# Annotations
ax_dg.text(dg_med, y_post - 0.42, f"median {dg_med:.2f}", ha="center",
           fontsize=fs_annot, color=PAL.ORANGE, fontweight="bold")
ax_dg.text(np.sqrt(dg_05 * dg_95), y_post + 0.40,
           f"90% CI: [{dg_05:.2f}, {dg_95:.1f}]",
           ha="center", fontsize=fs_ci, color="0.35")
ax_dg.text(np.sqrt(PRIOR_DG_LO * PRIOR_DG_HI), y_prior + 0.40,
           f"[{PRIOR_DG_LO:.2f}, {PRIOR_DG_HI:.1f}]",
           ha="center", fontsize=fs_ci, color="0.35")


# ── Suptitle ──
fig.suptitle(
    "Juno refit: prior vs posterior (35\u00b0 latitude, scenario-averaged)",
    fontsize=10 * SCALE, fontweight="bold", y=1.03,
)

fig.subplots_adjust(left=0.10, right=0.97, top=0.82, bottom=0.18, wspace=0.35)
out_dir = os.path.join(os.path.dirname(__file__), "..", "figures")
save_fig(fig, "midlat_juno_shrinkage", out_dir, formats=("png",))
