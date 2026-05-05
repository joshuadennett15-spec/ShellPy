"""
Plot Sobol sensitivity indices from run_sobol_suite.py results.

Produces three publication figures:
  1. Horizontal bar chart of total-order indices (ST) for thesis QoIs.
  2. S1 vs ST scatter showing interaction structure.
  3. Convergence of ST across checkpoint sample sizes.

Usage:
    python scripts/plot_sobol.py [--run-dir results/sobol/global_audited_sobol_params_N512]
"""
import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from pub_style import apply_style, PAL, figsize_single, figsize_double, label_panel, save_fig

FIGURES_DIR = Path(__file__).resolve().parents[1] / "figures" / "sobol"
SOBOL_DIR = Path(__file__).resolve().parents[1] / "results" / "sobol"

THESIS_QOIS = ["thickness_km", "D_cond_km", "D_conv_km", "lid_fraction"]
QOI_LABELS = {
    "thickness_km": "Total thickness (km)",
    "D_cond_km": "Conductive lid (km)",
    "D_conv_km": "Convective layer (km)",
    "lid_fraction": "Lid fraction",
    "convective_flag": "Convective fraction",
    "Ra": "Rayleigh number",
    "Nu": "Nusselt number",
    "valid_flag": "Valid flag",
    "physical_flag": "Physical flag",
}

FACTOR_SHORT = {
    "q_basal_target_mW_m2": "$q_{\\mathrm{basal}}$",
    "d_grain_mm": "$d_{\\mathrm{grain}}$",
    "epsilon_0": "$\\varepsilon_0$",
    "T_surf_K": "$T_{\\mathrm{surf}}$",
    "D_H2O_km": "$D_{\\mathrm{H_2O}}$",
    "mu_ice_GPa": "$\\mu_{\\mathrm{ice}}$",
    "Q_v_kJ_mol": "$Q_v$",
    "Q_b_kJ_mol": "$Q_b$",
    "H_rad_pW_kg": "$H_{\\mathrm{rad}}$",
    "f_porosity": "$f_{\\mathrm{por}}$",
    # grouped factors
    "basal_flux": "Basal flux",
    "shell_rheology": "Shell rheology",
    "shell_tides": "Shell tides",
    "surface_boundary": "Surface BC",
    "porosity": "Porosity",
}


def load_indices_csv(csv_path):
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def parse_rows(rows):
    """Parse CSV rows into nested dict: {output: {sample_size: {factor: {S1, S1_conf, ST, ST_conf}}}}."""
    data = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        if row["index_type"] != "main":
            continue
        output = row["output"]
        try:
            n = int(row["sample_size"])
        except (ValueError, TypeError):
            continue
        factor = row["factor"]
        try:
            st = float(row["ST"])
            st_conf = float(row["ST_conf"])
            s1 = float(row["S1"])
            s1_conf = float(row["S1_conf"])
        except (ValueError, TypeError):
            continue
        if np.isnan(st):
            continue
        data[output][n][factor] = {
            "S1": s1, "S1_conf": s1_conf,
            "ST": st, "ST_conf": st_conf,
        }
    return data


def _factor_label(name):
    return FACTOR_SHORT.get(name, name)


def plot_tornado(data, run_label, qois=None):
    """Horizontal bar chart of ST at final N for each thesis QoI."""
    qois = qois or [q for q in THESIS_QOIS if q in data]
    n_qois = len(qois)
    if n_qois == 0:
        return None

    fig, axes = plt.subplots(1, n_qois, figsize=(figsize_double()[0], 3.2),
                             sharey=False, constrained_layout=True)
    if n_qois == 1:
        axes = [axes]

    letters = "abcdefgh"
    colours = [PAL.BLUE, PAL.ORANGE, PAL.GREEN, PAL.RED, PAL.PURPLE,
               PAL.CYAN, PAL.YELLOW, PAL.BLACK, "#888888", "#AAAAAA"]

    for idx, (ax, qoi) in enumerate(zip(axes, qois)):
        final_n = max(data[qoi].keys())
        factors_data = data[qoi][final_n]

        sorted_factors = sorted(factors_data.items(), key=lambda x: x[1]["ST"], reverse=True)
        names = [_factor_label(f) for f, _ in sorted_factors]
        st_vals = [v["ST"] for _, v in sorted_factors]
        st_errs = [v["ST_conf"] for _, v in sorted_factors]

        y_pos = np.arange(len(names))
        ax.barh(y_pos, st_vals, xerr=st_errs, height=0.6,
                color=colours[:len(names)], edgecolor="none",
                capsize=2, error_kw={"lw": 0.8})
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names)
        ax.set_xlabel("$S_T$")
        ax.set_title(f"({letters[idx]})  {QOI_LABELS.get(qoi, qoi)}", fontweight="bold")
        ax.set_xlim(0, min(1.15, max(st_vals) * 1.4))
        ax.axvline(0.05, color="0.6", ls=":", lw=0.6)
        ax.invert_yaxis()

    return fig


def plot_s1_vs_st(data, run_label, qois=None):
    """S1 vs ST scatter — distance from diagonal shows interaction strength."""
    qois = qois or [q for q in THESIS_QOIS if q in data]
    n_qois = len(qois)
    if n_qois == 0:
        return None

    fig, axes = plt.subplots(1, n_qois, figsize=(figsize_double()[0], 3.4),
                             sharex=True, sharey=True, constrained_layout=True)
    if n_qois == 1:
        axes = [axes]

    letters = "abcdefgh"
    for idx, (ax, qoi) in enumerate(zip(axes, qois)):
        final_n = max(data[qoi].keys())
        factors_data = data[qoi][final_n]

        for factor, vals in factors_data.items():
            ax.errorbar(vals["ST"], vals["S1"],
                        xerr=vals["ST_conf"], yerr=vals["S1_conf"],
                        fmt="o", ms=4, capsize=2, lw=0.8,
                        label=_factor_label(factor))

        lim = max(ax.get_xlim()[1], ax.get_ylim()[1], 0.5)
        ax.plot([0, lim], [0, lim], "k--", lw=0.5, alpha=0.4, zorder=0)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_xlabel("$S_T$ (total)")
        if idx == 0:
            ax.set_ylabel("$S_1$ (first-order)")
        ax.set_title(f"({letters[idx]})  {QOI_LABELS.get(qoi, qoi)}", fontweight="bold")

    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center",
               ncol=min(5, len(labels)), fontsize=6.5)
    return fig


def plot_convergence(data, run_label, qois=None, top_n=4):
    """ST convergence across sample sizes for top factors."""
    qois = qois or [q for q in THESIS_QOIS if q in data]
    n_qois = len(qois)
    if n_qois == 0:
        return None

    fig, axes = plt.subplots(1, n_qois, figsize=(figsize_double()[0], 3.2),
                             sharey=False, constrained_layout=True)
    if n_qois == 1:
        axes = [axes]

    letters = "abcdefgh"
    line_colours = [PAL.BLUE, PAL.ORANGE, PAL.GREEN, PAL.RED, PAL.PURPLE]

    for idx, (ax, qoi) in enumerate(zip(axes, qois)):
        sample_sizes = sorted(data[qoi].keys())
        if len(sample_sizes) < 2:
            ax.text(0.5, 0.5, "Single checkpoint\n(no convergence)",
                    ha="center", va="center", transform=ax.transAxes, fontsize=7)
            ax.set_title(f"({letters[idx]})  {QOI_LABELS.get(qoi, qoi)}", fontweight="bold")
            continue

        final_n = max(sample_sizes)
        top_factors = sorted(data[qoi][final_n].items(),
                             key=lambda x: x[1]["ST"], reverse=True)[:top_n]

        for ci, (factor, _) in enumerate(top_factors):
            ns = []
            sts = []
            errs = []
            for n in sample_sizes:
                if factor in data[qoi][n]:
                    ns.append(n)
                    sts.append(data[qoi][n][factor]["ST"])
                    errs.append(data[qoi][n][factor]["ST_conf"])
            ax.errorbar(ns, sts, yerr=errs, fmt="o-", ms=3, capsize=2,
                        lw=1.0, color=line_colours[ci % len(line_colours)],
                        label=_factor_label(factor))

        ax.set_xlabel("$N$ (base samples)")
        if idx == 0:
            ax.set_ylabel("$S_T$")
        ax.set_title(f"({letters[idx]})  {QOI_LABELS.get(qoi, qoi)}", fontweight="bold")
        ax.legend(fontsize=6, loc="best")

    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="", help="Path to a specific Sobol run directory.")
    parser.add_argument("--all", action="store_true", help="Plot all runs found in results/sobol/.")
    parser.add_argument("--qois", default="", help="Comma-separated QoI list (default: thesis set).")
    parser.add_argument("--formats", default="png,pdf", help="Output formats.")
    args = parser.parse_args()

    apply_style()
    formats = tuple(f.strip() for f in args.formats.split(","))
    qois = [q.strip() for q in args.qois.split(",") if q.strip()] or None

    if args.run_dir:
        run_dirs = [Path(args.run_dir)]
    elif args.all:
        run_dirs = sorted(p for p in SOBOL_DIR.iterdir() if p.is_dir())
    else:
        run_dirs = sorted(p for p in SOBOL_DIR.iterdir() if p.is_dir())

    for run_dir in run_dirs:
        csv_files = list(run_dir.glob("*_indices.csv"))
        if not csv_files:
            print(f"  Skipping {run_dir.name}: no indices CSV found")
            continue

        csv_path = csv_files[0]
        run_label = run_dir.name
        print(f"\nPlotting: {run_label}")

        rows = load_indices_csv(csv_path)
        data = parse_rows(rows)

        out_dir = FIGURES_DIR / run_label
        out_dir.mkdir(parents=True, exist_ok=True)

        fig = plot_tornado(data, run_label, qois)
        if fig:
            save_fig(fig, f"{run_label}_tornado", str(out_dir), formats)

        fig = plot_s1_vs_st(data, run_label, qois)
        if fig:
            save_fig(fig, f"{run_label}_s1_vs_st", str(out_dir), formats)

        fig = plot_convergence(data, run_label, qois)
        if fig:
            save_fig(fig, f"{run_label}_convergence", str(out_dir), formats)


if __name__ == "__main__":
    main()
