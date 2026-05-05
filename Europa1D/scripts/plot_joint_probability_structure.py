#!/usr/bin/env python3
"""
Howell-style four-panel joint-density plots for shell-structure Monte Carlo runs.

This script renders one figure per input NPZ file using a publication-style
layout inspired by the classic conductive-vs-convective thickness plots:
  (a) D_cond vs D_conv joint density
  (b) zoomed view near the total-thickness CBE
  (c) H_total vs D_cond joint density
  (d) H_total vs D_conv joint density

By default the script focuses on the convective branch, matching the original
Howell-style diagnostic. Use --all-samples to include conductive-only draws.

If no inputs are provided, the script auto-discovers the newer latitude files
available in this repo:
  - results/endmember_*_andrade.npz
  - results/midlat_juno/midlat35_*_constrained.npz

For the midlat35 constrained files, remember these are Juno-conditioned
posteriors rather than unconstrained prior draws.

Usage:
    python plot_joint_probability_structure.py
    python plot_joint_probability_structure.py endmember_uniform_eq_andrade
    python plot_joint_probability_structure.py results/endmember_uniform_eq_andrade.npz
    python plot_joint_probability_structure.py --all-samples midlat35_uniform_constrained
"""
import argparse
import glob
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS_DIR)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from matplotlib.patches import Rectangle
from scipy.ndimage import gaussian_filter
from scipy.stats import gaussian_kde

from pub_style import apply_style, figsize_double_tall, save_fig

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures", "pub")
RA_CRIT = 1.0e3

apply_style()


def _pretty_case_label(stem: str) -> str:
    """Human-readable title from a result stem."""
    label = stem
    replacements = [
        ("endmember_", ""),
        ("midlat35_", "Midlat 35 deg "),
        ("_andrade", ""),
        ("_constrained", " constrained posterior"),
        ("_round1", " round 1"),
        ("_round2", " round 2"),
        ("_round3", " round 3"),
        ("_round4", " round 4"),
        ("_round5", " round 5"),
        ("lemasquerier", "Lemasquerier"),
        ("soderlund", "Soderlund"),
        ("uniform", "Uniform"),
        ("pole", "Pole"),
        ("mid", "Mid"),
        ("eq", "Eq"),
    ]
    for old, new in replacements:
        label = label.replace(old, new)
    label = label.replace("_", " ")
    return " ".join(label.split())


def _discover_inputs() -> list[str]:
    """Auto-discover recent endmember and mid-latitude files."""
    patterns = [
        os.path.join(RESULTS_DIR, "endmember_*_andrade.npz"),
        os.path.join(RESULTS_DIR, "midlat_juno", "midlat35_*_constrained.npz"),
    ]
    paths: list[str] = []
    for pattern in patterns:
        paths.extend(sorted(glob.glob(pattern)))

    if paths:
        return paths

    fallbacks = [
        os.path.join(RESULTS_DIR, "mc_15000_optionA_v2_andrade.npz"),
        os.path.join(RESULTS_DIR, "global_updated_ptidal.npz"),
        os.path.join(RESULTS_DIR, "monte_carlo_results.npz"),
    ]
    return [path for path in fallbacks if os.path.exists(path)]


def _resolve_inputs(inputs: list[str]) -> list[str]:
    """Resolve user-provided paths or bare stems to actual NPZ files."""
    if not inputs:
        return _discover_inputs()

    resolved: list[str] = []
    candidates: list[str]
    for item in inputs:
        candidates = [
            item,
            f"{item}.npz",
            os.path.join(RESULTS_DIR, item),
            os.path.join(RESULTS_DIR, f"{item}.npz"),
            os.path.join(RESULTS_DIR, "midlat_juno", item),
            os.path.join(RESULTS_DIR, "midlat_juno", f"{item}.npz"),
        ]
        found = next((path for path in candidates if os.path.exists(path)), None)
        if found is not None:
            resolved.append(os.path.abspath(found))
            continue

        glob_hits = sorted(
            glob.glob(os.path.join(RESULTS_DIR, "**", f"{item}*.npz"), recursive=True)
        )
        if glob_hits:
            resolved.extend(os.path.abspath(path) for path in glob_hits)
            continue

        print(f"WARNING: could not resolve input '{item}'")

    deduped: list[str] = []
    seen: set[str] = set()
    for path in resolved:
        if path not in seen:
            deduped.append(path)
            seen.add(path)
    return deduped


def _safe_mode(values: np.ndarray) -> float:
    """Mode estimate via KDE, with robust fallbacks."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return 0.0
    if len(values) < 8 or np.std(values) < 1.0e-6:
        return float(np.median(values))

    lo = max(0.0, float(np.percentile(values, 0.2)) - 1.0)
    hi = float(np.percentile(values, 99.8)) + 1.0
    if hi <= lo:
        return float(np.median(values))

    grid = np.linspace(lo, hi, 512)
    try:
        kde = gaussian_kde(values)
        return float(grid[np.argmax(kde(grid))])
    except (np.linalg.LinAlgError, ValueError):
        return float(np.median(values))


def _select_h_cbe_window(H: np.ndarray, h_cbe: float, min_samples: int) -> tuple[np.ndarray, float]:
    """Expand a symmetric window around the total-thickness CBE until populated."""
    width = max(3.0, 0.06 * h_cbe)
    h_span = max(10.0, float(np.percentile(H, 99.0) - np.percentile(H, 1.0)))
    max_width = max(width * 2.0, 0.35 * h_span)

    mask = np.abs(H - h_cbe) <= width
    while mask.sum() < min_samples and width < max_width:
        width *= 1.35
        mask = np.abs(H - h_cbe) <= width
    return mask, width


def _nice_upper(values: np.ndarray, floor: float) -> float:
    """Rounded upper axis bound from the high-percentile tail."""
    upper = max(floor, float(np.percentile(values, 99.5)) * 1.05)
    return float(5.0 * np.ceil(upper / 5.0))


def _joint_density(
    x: np.ndarray,
    y: np.ndarray,
    xlim: float,
    ylim: float,
    bins: int = 220,
    sigma: float = 1.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Smoothed 2D histogram density."""
    hist, x_edges, y_edges = np.histogram2d(
        x,
        y,
        bins=[bins, bins],
        range=[[0.0, xlim], [0.0, ylim]],
        density=True,
    )
    hist = gaussian_filter(hist, sigma=sigma, mode="nearest")
    return hist, x_edges, y_edges


def _panel_tag(ax, letter: str) -> None:
    """White in-panel label matching the Howell-style layout."""
    ax.text(
        0.03,
        0.97,
        f"({letter})",
        transform=ax.transAxes,
        ha="left",
        va="top",
        color="white",
        fontsize=9,
        fontweight="bold",
    )


def _apply_heatmap_axes_style(ax) -> None:
    """Black panel background with readable ticks."""
    ax.set_facecolor("black")
    ax.tick_params(which="both", color="white", labelcolor="black")


def _draw_heatmap(ax, density: np.ndarray, x_edges: np.ndarray, y_edges: np.ndarray, cmap, norm):
    """Render a smoothed density grid."""
    image = ax.imshow(
        density.T,
        origin="lower",
        extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
        aspect="auto",
        cmap=cmap,
        norm=norm,
        interpolation="bilinear",
    )
    return image


def _load_case(path: str, include_all_samples: bool) -> dict:
    """Load and optionally filter a Monte Carlo archive."""
    data = np.load(path)
    H = np.asarray(data["thicknesses_km"], dtype=float)
    D_cond = np.asarray(data["D_cond_km"], dtype=float)
    D_conv = np.asarray(data["D_conv_km"], dtype=float)
    Ra = (
        np.asarray(data["Ra_values"], dtype=float)
        if "Ra_values" in data
        else np.full(len(H), np.nan, dtype=float)
    )

    valid = np.isfinite(H) & np.isfinite(D_cond) & np.isfinite(D_conv)
    H = H[valid]
    D_cond = D_cond[valid]
    D_conv = D_conv[valid]
    Ra = Ra[valid]

    conv_mask = D_conv > 0.5
    if np.isfinite(Ra).any():
        conv_mask &= np.where(np.isfinite(Ra), Ra >= RA_CRIT, True)

    used_convective_subset = False
    if include_all_samples:
        use_mask = np.ones(len(H), dtype=bool)
    elif conv_mask.sum() >= 50:
        use_mask = conv_mask
        used_convective_subset = True
    else:
        use_mask = np.ones(len(H), dtype=bool)

    H_use = H[use_mask]
    D_cond_use = D_cond[use_mask]
    D_conv_use = D_conv[use_mask]

    h_cbe = _safe_mode(H_use)
    min_samples = max(40, int(0.025 * len(H_use)))
    window_mask, width = _select_h_cbe_window(H_use, h_cbe, min_samples)

    cond_window = D_cond_use[window_mask]
    conv_window = D_conv_use[window_mask]
    conv_window_pos = conv_window[conv_window > 0.5]
    if len(conv_window_pos) < 8:
        conv_window_pos = D_conv_use[D_conv_use > 0.5]

    cond_cbe = _safe_mode(cond_window)
    conv_cbe = _safe_mode(conv_window_pos)

    return {
        "path": path,
        "stem": Path(path).stem,
        "label": _pretty_case_label(Path(path).stem),
        "H": H_use,
        "D_cond": D_cond_use,
        "D_conv": D_conv_use,
        "n_loaded": len(H),
        "n_used": len(H_use),
        "used_convective_subset": used_convective_subset,
        "h_cbe": h_cbe,
        "cond_cbe": cond_cbe,
        "conv_cbe": conv_cbe,
        "window_mask": window_mask,
        "window_width": width,
    }


def _plot_case(case: dict, output_dir: str) -> None:
    """Render one four-panel figure."""
    H = case["H"]
    D_cond = case["D_cond"]
    D_conv = case["D_conv"]
    h_cbe = case["h_cbe"]
    cond_cbe = case["cond_cbe"]
    conv_cbe = case["conv_cbe"]

    x_max = _nice_upper(D_cond, 25.0)
    y_max = _nice_upper(D_conv, 25.0)
    h_max = _nice_upper(H, max(35.0, h_cbe + 5.0))
    zoom_max = min(
        max(x_max, y_max),
        _nice_upper(np.concatenate([D_cond, D_conv]), max(25.0, h_cbe + 5.0)),
    )

    panels = {
        "a": _joint_density(D_cond, D_conv, x_max, y_max),
        "b": _joint_density(D_cond, D_conv, zoom_max, zoom_max),
        "c": _joint_density(H, D_cond, h_max, x_max),
        "d": _joint_density(H, D_conv, h_max, y_max),
    }

    positive = np.concatenate(
        [grid[grid > 0.0] for grid, _, _ in panels.values() if np.any(grid > 0.0)]
    )
    if len(positive) == 0:
        print(f"  WARNING: no positive density for {case['stem']}, skipping")
        return

    vmin = float(np.percentile(positive, 8.0))
    vmax = float(np.percentile(positive, 99.7))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin = float(np.min(positive))
        vmax = float(np.max(positive))
    if vmax <= vmin:
        vmax = vmin * 1.01

    cmap = plt.get_cmap("inferno").copy()
    cmap.set_under("black")
    cmap.set_bad("black")
    norm = PowerNorm(gamma=0.55, vmin=vmin, vmax=vmax)

    fig = plt.figure(figsize=figsize_double_tall(1.02))
    gs = fig.add_gridspec(
        2,
        3,
        width_ratios=[1.0, 1.95, 0.11],
        height_ratios=[1.0, 1.0],
        wspace=0.18,
        hspace=0.28,
    )

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    cax = fig.add_subplot(gs[0, 2])

    for ax in (ax_a, ax_b, ax_c, ax_d):
        _apply_heatmap_axes_style(ax)

    im = _draw_heatmap(ax_a, *panels["a"], cmap=cmap, norm=norm)
    _draw_heatmap(ax_b, *panels["b"], cmap=cmap, norm=norm)
    _draw_heatmap(ax_c, *panels["c"], cmap=cmap, norm=norm)
    _draw_heatmap(ax_d, *panels["d"], cmap=cmap, norm=norm)

    ax_a.set_xlim(0, x_max)
    ax_a.set_ylim(0, y_max)
    ax_b.set_xlim(0, zoom_max)
    ax_b.set_ylim(0, zoom_max)
    ax_c.set_xlim(0, h_max)
    ax_c.set_ylim(0, x_max)
    ax_d.set_xlim(0, h_max)
    ax_d.set_ylim(0, y_max)

    # Constant total-thickness guide in the layer-thickness panels.
    x_line_full = np.linspace(0.0, min(x_max, h_cbe), 300)
    y_line_full = h_cbe - x_line_full
    y_line_full = np.clip(y_line_full, 0.0, y_max)
    ax_a.plot(x_line_full, y_line_full, ls="--", lw=0.7, color="white", alpha=0.75)

    x_line_zoom = np.linspace(0.0, min(zoom_max, h_cbe), 300)
    y_line_zoom = np.clip(h_cbe - x_line_zoom, 0.0, zoom_max)
    ax_b.plot(x_line_zoom, y_line_zoom, ls="--", lw=0.8, color="white", alpha=0.85)

    ax_a.add_patch(
        Rectangle(
            (0.0, 0.0),
            zoom_max,
            zoom_max,
            fill=False,
            ec="white",
            lw=0.8,
        )
    )
    ax_a.text(
        zoom_max * 0.72,
        zoom_max * 0.11,
        "(b)",
        color="white",
        fontsize=8,
        ha="left",
        va="bottom",
    )

    for ax in (ax_a, ax_b):
        ax.scatter(
            [cond_cbe],
            [conv_cbe],
            s=30,
            facecolor="white",
            edgecolor="black",
            linewidth=0.4,
            zorder=6,
        )

    for ax, y_val in ((ax_c, cond_cbe), (ax_d, conv_cbe)):
        ax.scatter(
            [h_cbe],
            [y_val],
            s=24,
            facecolor="white",
            edgecolor="black",
            linewidth=0.4,
            zorder=6,
        )

    text_box = (
        f"{case['label']}\n"
        f"N used = {case['n_used']:,}\n"
        f"H CBE = {h_cbe:.1f} km"
    )
    if case["used_convective_subset"]:
        text_box += "\nconvective branch only"
    if "constrained" in case["stem"]:
        text_box += "\nJuno-conditioned posterior"

    ax_a.text(
        0.04,
        0.08,
        text_box,
        transform=ax_a.transAxes,
        color="white",
        fontsize=6.3,
        ha="left",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.28", fc=(0, 0, 0, 0.55), ec="white", lw=0.4),
    )

    ax_b.text(
        0.53,
        0.49,
        f"CBE = {h_cbe:.1f} km total thickness",
        transform=ax_b.transAxes,
        color="black",
        fontsize=7,
        rotation=-38,
        ha="center",
        va="center",
    )

    ax_c.axvline(h_cbe, color="white", ls="--", lw=0.6, alpha=0.75)
    ax_d.axvline(h_cbe, color="white", ls="--", lw=0.6, alpha=0.75)

    ax_a.set_xlabel("Conductive Thickness [km]")
    ax_a.set_ylabel("Convective Thickness [km]")
    ax_b.set_xlabel("Conductive Thickness [km]")
    ax_b.set_ylabel("Convective Thickness [km]")
    ax_c.set_xlabel("Total Thickness [km]")
    ax_c.set_ylabel("Conductive Thickness [km]")
    ax_d.set_xlabel("Total Thickness [km]")
    ax_d.set_ylabel("Convective Thickness [km]")

    for ax, letter in zip((ax_a, ax_b, ax_c, ax_d), "abcd"):
        _panel_tag(ax, letter)

    cbar = fig.colorbar(im, cax=cax)
    cbar.set_ticks([])
    cbar.outline.set_linewidth(0.4)
    cax.text(
        1.8,
        0.98,
        "More Likely",
        rotation=90,
        transform=cax.transAxes,
        ha="center",
        va="top",
        fontsize=7,
    )
    cax.text(
        1.8,
        0.02,
        "Less Likely",
        rotation=90,
        transform=cax.transAxes,
        ha="center",
        va="bottom",
        fontsize=7,
    )

    fig.suptitle(
        f"Joint shell-structure density: {case['label']}",
        fontsize=9,
        y=0.995,
    )

    output_name = f"fig_jointprob_{case['stem']}"
    save_fig(fig, output_name, output_dir)
    print(
        f"  {case['stem']}: loaded={case['n_loaded']:,}, used={case['n_used']:,}, "
        f"H_CBE={h_cbe:.1f} km, D_cond_CBE={cond_cbe:.1f} km, D_conv_CBE={conv_cbe:.1f} km, "
        f"window=+/-{case['window_width']:.1f} km"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create Howell-style four-panel joint-density shell-structure plots."
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        help="NPZ file paths or bare result stems. Defaults to auto-discovered recent latitude files.",
    )
    parser.add_argument(
        "--all-samples",
        action="store_true",
        help="Include conductive-only draws instead of focusing on the convective branch.",
    )
    args = parser.parse_args()

    paths = _resolve_inputs(args.inputs)
    if not paths:
        print("No matching input NPZ files found.")
        return 1

    os.makedirs(FIGURES_DIR, exist_ok=True)
    print("Joint-density inputs:")
    for path in paths:
        print(f"  - {os.path.relpath(path, RESULTS_DIR)}")

    for path in paths:
        try:
            case = _load_case(path, include_all_samples=args.all_samples)
            _plot_case(case, FIGURES_DIR)
        except Exception as exc:  # pragma: no cover - best-effort batch plotting
            print(f"  ERROR while plotting {path}: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
