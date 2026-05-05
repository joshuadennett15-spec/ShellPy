"""
Two-panel figure: Bayes factor vs constraint latitude and N_eff/N vs latitude.

Uses Europa2D/results/bayes_factor_latitude_sweep.npz produced by
bayes_factor_latitude_sweep.py.
"""
from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.join(_SCRIPT_DIR, "..")
sys.path.insert(0, _SCRIPT_DIR)
from pub_style import apply_style, figsize_double  # noqa: E402

RESULTS_NPZ = os.path.join(_PROJECT_DIR, "results", "bayes_factor_latitude_sweep.npz")
OUTPUT_DIR = os.path.join(_PROJECT_DIR, "figures", "thesis")
OUTPUT_PDF = os.path.join(OUTPUT_DIR, "fig_bayes_factor_latitude.pdf")
OUTPUT_PNG = os.path.join(OUTPUT_DIR, "fig_bayes_factor_latitude.png")

# Scenario colours (Okabe-Ito-inspired; colour-blind safe)
SCENARIO_COLOURS = {
    "Uniform":          "#0072B2",
    "Equator-enhanced": "#D55E00",
    "Polar-enhanced":   "#009E73",
    "Strong polar":     "#CC79A7",
}

PAIR_COLOURS = [
    "#0072B2", "#D55E00", "#009E73",
    "#CC79A7", "#56B4E9", "#E69F00",
]


def main() -> None:
    apply_style()

    d = np.load(RESULTS_NPZ, allow_pickle=True)
    lats = d["latitudes_deg"]
    bf = d["bayes_factor"]
    pairs = d["pair_indices"]
    labels = [str(x) for x in d["scenario_labels"]]
    n_eff = d["n_eff"]
    n_samples = int(d["n_samples"])
    juno_sigma = float(d["juno_sigma_eff_km"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize_double(aspect=0.50))

    # ----- Panel (a): log10 BF vs lat for each pair -----
    for p, (a, b) in enumerate(pairs):
        pair_lbl = f"{labels[a]} / {labels[b]}"
        ax1.plot(lats, np.log10(bf[p]), color=PAIR_COLOURS[p], lw=1.6,
                 label=pair_lbl)

    # Shade the "substantial evidence" band (|log10 BF| >= log10 3)
    ymin, ymax = -0.65, 0.65
    ax1.axhspan(np.log10(3.0), ymax, color="0.85", alpha=0.5, zorder=0)
    ax1.axhspan(ymin, -np.log10(3.0), color="0.85", alpha=0.5, zorder=0)
    ax1.axhline( np.log10(3.0), color="k", lw=0.8, ls="--", alpha=0.6)
    ax1.axhline(-np.log10(3.0), color="k", lw=0.8, ls="--", alpha=0.6)
    ax1.text(89, np.log10(3.0) + 0.015, r"BF $= 3$ (substantial)",
             ha="right", fontsize=7, color="0.25")
    ax1.text(89, -np.log10(3.0) - 0.06, r"BF $= 1/3$",
             ha="right", fontsize=7, color="0.25")

    # Juno-applied latitude marker
    ax1.axvline(35.0, color="0.3", lw=0.9, ls="-", alpha=0.5)
    ax1.text(35.6, 0.56, r"Juno ($35^\circ$)", fontsize=8, color="0.3")

    # "no discrimination" callout
    ax1.text(45.0, 0.23,
             "No pair crosses $\\mathrm{BF}=3$ at any latitude",
             fontsize=8, color="0.15", ha="center",
             bbox=dict(facecolor="white", edgecolor="0.6",
                       boxstyle="round,pad=0.25", lw=0.6))

    ax1.axhline(0.0, color="k", lw=0.5, alpha=0.4)
    ax1.set_xlabel(r"Constraint latitude $\phi$ (deg)")
    ax1.set_ylabel(r"$\log_{10}\, \mathrm{BF}(\mathcal{S}_a : \mathcal{S}_b)$")
    ax1.set_title("(a)  Scenario Bayes factor vs. Juno-applied latitude")
    ax1.set_xlim(0, 90)
    ax1.set_ylim(ymin, ymax)
    ax1.legend(loc="lower center", ncol=2, fontsize=7, frameon=False,
               handlelength=1.6, columnspacing=1.0)

    # ----- Panel (b): N_eff / N vs lat per scenario -----
    for i, label in enumerate(labels):
        ax2.plot(lats, n_eff[i] / n_samples,
                 color=SCENARIO_COLOURS.get(label, f"C{i}"), lw=1.8, label=label)

    ax2.axvline(35.0, color="0.3", lw=0.9, ls="-", alpha=0.5)
    ax2.text(35.5, 0.98, "Juno ($35^\\circ$)", fontsize=8, color="0.3")
    ax2.set_xlabel(r"Constraint latitude $\phi$ (deg)")
    ax2.set_ylabel(r"$N_{\mathrm{eff}} \, /\, N$")
    ax2.set_title(rf"(b)  Informativeness of a Juno-type constraint "
                  rf"($\sigma_{{\mathrm{{eff}}}} = {juno_sigma:.1f}$~km)")
    ax2.set_xlim(0, 90)
    ax2.set_ylim(0.5, 1.0)
    ax2.legend(loc="lower right", fontsize=7, frameon=False, handlelength=1.6)

    fig.tight_layout()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig.savefig(OUTPUT_PDF, bbox_inches="tight")
    fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    print(f"Wrote {OUTPUT_PDF}")
    print(f"Wrote {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
