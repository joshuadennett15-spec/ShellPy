"""
Run paired Maxwell vs Andrade Monte Carlo ensembles (5000 each)
and generate a rheology comparison figure for the results chapter.

Usage:
    python scripts/run_rheology_comparison.py [--n-workers 14]
"""
import argparse
import json
import os
import sys
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(SCRIPT_DIR, "..")
SRC_DIR = os.path.join(PROJECT_DIR, "src")
CONFIG_PATH = os.path.join(SRC_DIR, "config.json")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
FIGURES_DIR = os.path.join(PROJECT_DIR, "figures", "pub")

N_ITER = 5000
SEED = 44


def _set_rheology_model(model: str) -> dict:
    """Set the rheology model in config.json and return the old config."""
    with open(CONFIG_PATH, "r") as f:
        cfg = json.load(f)
    old_model = cfg.get("rheology", {}).get("model", "Maxwell")
    cfg["rheology"]["model"] = model
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=4)
    print(f"  config.json rheology.model: {old_model} -> {model}")
    return cfg


def _run_mc(label: str, n_workers: int) -> str:
    """Run MC via subprocess and return the output path."""
    out_name = f"mc_rheology_{label.lower()}_{N_ITER}.npz"
    out_path = os.path.join(RESULTS_DIR, out_name)

    print(f"\n{'='*60}")
    print(f"  Running {label} MC: {N_ITER} iterations, seed={SEED}")
    print(f"  Output: {out_path}")
    print(f"{'='*60}\n")

    # Run via subprocess so constants.py re-reads the modified config.json
    # Use replace to avoid backslash issues in f-strings on Windows
    src_escaped = SRC_DIR.replace("\\", "/")
    out_escaped = out_path.replace("\\", "/")
    runner_code = (
        "import sys, os\n"
        f"sys.path.insert(0, '{src_escaped}')\n"
        "import multiprocessing as mp\n"
        "mp.freeze_support()\n"
        "from Monte_Carlo import MonteCarloRunner, save_results\n"
        f"runner = MonteCarloRunner(n_iterations={N_ITER}, seed={SEED}, n_workers={n_workers})\n"
        "results = runner.run()\n"
        f"save_results(results, '{out_escaped}')\n"
        f"print('Saved: {out_escaped}')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", runner_code],
        cwd=SRC_DIR,
        capture_output=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"{label} MC run failed with code {result.returncode}")

    return out_path


def _generate_figure(maxwell_path: str, andrade_path: str):
    """Generate the Maxwell vs Andrade comparison figure."""
    import numpy as np

    sys.path.insert(0, os.path.join(PROJECT_DIR, "..", "Europa2D", "scripts"))
    sys.path.insert(0, SRC_DIR)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    # Minimal pub style
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 8,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "axes.titleweight": "bold",
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })

    m = np.load(maxwell_path, allow_pickle=True)
    a = np.load(andrade_path, allow_pickle=True)

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8))
    fig.subplots_adjust(wspace=0.38)

    C_MAX = "0.55"
    C_AND = "#0072B2"

    # (a) H_total distributions
    ax = axes[0]
    bins = np.linspace(5, 80, 60)
    ax.hist(m["thicknesses_km"], bins=bins, density=True, alpha=0.5,
            color=C_MAX, edgecolor="k", linewidth=0.3, label="Maxwell")
    ax.hist(a["thicknesses_km"], bins=bins, density=True, alpha=0.5,
            color=C_AND, edgecolor="k", linewidth=0.3, label="Andrade")
    ax.axvline(np.median(m["thicknesses_km"]), color=C_MAX, ls="--", lw=1.2)
    ax.axvline(np.median(a["thicknesses_km"]), color=C_AND, ls="--", lw=1.2)
    ax.set_xlabel(r"$H_{\rm total}$ (km)")
    ax.set_ylabel("Density")
    ax.set_title("Total shell thickness")
    ax.legend(fontsize=6)
    ax.text(0.97, 0.92,
            f"Maxwell: {np.median(m['thicknesses_km']):.1f} km\n"
            f"Andrade: {np.median(a['thicknesses_km']):.1f} km",
            transform=ax.transAxes, fontsize=6, ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.9, ec="0.7", lw=0.3))
    ax.text(-0.12, 1.06, "(a)", transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="bottom")

    # (b) D_cond distributions
    ax = axes[1]
    ax.hist(m["D_cond_km"], bins=bins, density=True, alpha=0.5,
            color=C_MAX, edgecolor="k", linewidth=0.3, label="Maxwell")
    ax.hist(a["D_cond_km"], bins=bins, density=True, alpha=0.5,
            color=C_AND, edgecolor="k", linewidth=0.3, label="Andrade")
    ax.axvline(np.median(m["D_cond_km"]), color=C_MAX, ls="--", lw=1.2)
    ax.axvline(np.median(a["D_cond_km"]), color=C_AND, ls="--", lw=1.2)
    ax.set_xlabel(r"$D_{\rm cond}$ (km)")
    ax.set_ylabel("Density")
    ax.set_title("Conductive lid thickness")
    ax.text(0.97, 0.92,
            f"Maxwell: {np.median(m['D_cond_km']):.1f} km\n"
            f"Andrade: {np.median(a['D_cond_km']):.1f} km",
            transform=ax.transAxes, fontsize=6, ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.9, ec="0.7", lw=0.3))
    ax.text(-0.12, 1.06, "(b)", transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="bottom")

    # (c) Convecting fraction bar comparison
    ax = axes[2]
    conv_m = np.sum(m["Nu_values"] > 1.1) / len(m["Nu_values"]) * 100
    conv_a = np.sum(a["Nu_values"] > 1.1) / len(a["Nu_values"]) * 100
    bars = ax.bar(["Maxwell", "Andrade"], [conv_m, conv_a],
                  color=[C_MAX, C_AND], edgecolor="k", linewidth=0.5, alpha=0.7)
    for bar, v in zip(bars, [conv_m, conv_a]):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 1,
                f"{v:.0f}%", ha="center", va="bottom", fontsize=7, fontweight="bold")
    ax.set_ylabel("Convecting realisations (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Convective fraction")
    ax.text(-0.12, 1.06, "(c)", transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="bottom")

    os.makedirs(FIGURES_DIR, exist_ok=True)
    for fmt in ("png", "pdf"):
        path = os.path.join(FIGURES_DIR, f"fig_rheology_comparison.{fmt}")
        fig.savefig(path, dpi=300 if fmt == "png" else None,
                    transparent=(fmt == "pdf"))
    plt.close(fig)
    print(f"\nSaved: fig_rheology_comparison.{{png, pdf}} -> {FIGURES_DIR}")


def main():
    parser = argparse.ArgumentParser(description="Maxwell vs Andrade MC comparison")
    parser.add_argument("--n-workers", type=int, default=14)
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Save original config
    with open(CONFIG_PATH, "r") as f:
        original_cfg = json.load(f)

    try:
        # Run Maxwell
        _set_rheology_model("Maxwell")
        maxwell_path = _run_mc("Maxwell", args.n_workers)

        # Run Andrade
        _set_rheology_model("Andrade")
        andrade_path = _run_mc("Andrade", args.n_workers)

    finally:
        # Always restore original config
        with open(CONFIG_PATH, "w") as f:
            json.dump(original_cfg, f, indent=4)
        print(f"\nRestored config.json to original state (model={original_cfg['rheology']['model']})")

    # Generate comparison figure
    print("\nGenerating comparison figure...")
    _generate_figure(maxwell_path, andrade_path)

    # Print summary
    import numpy as np
    m = np.load(maxwell_path, allow_pickle=True)
    a = np.load(andrade_path, allow_pickle=True)

    print(f"\n{'='*60}")
    print(f"  Rheology Comparison Summary ({N_ITER} MC each)")
    print(f"{'='*60}")
    print(f"  {'Metric':<25s}  {'Maxwell':>10s}  {'Andrade':>10s}  {'Delta':>10s}")
    print(f"  {'':-<25s}  {'':->10s}  {'':->10s}  {'':->10s}")

    for label, key in [("H_total median (km)", "thicknesses_km"),
                       ("D_cond median (km)", "D_cond_km")]:
        vm = np.median(m[key])
        va = np.median(a[key])
        print(f"  {label:<25s}  {vm:10.1f}  {va:10.1f}  {va-vm:+10.1f}")

    conv_m = np.sum(m["Nu_values"] > 1.1) / len(m["Nu_values"]) * 100
    conv_a = np.sum(a["Nu_values"] > 1.1) / len(a["Nu_values"]) * 100
    print(f"  {'Conv fraction (%)':<25s}  {conv_m:10.0f}  {conv_a:10.0f}  {conv_a-conv_m:+10.0f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
