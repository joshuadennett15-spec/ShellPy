import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

from Convection import IceConvection
from constants import Thermal

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')


from scipy.stats import gaussian_kde

def _get_kde_pdf(values: np.ndarray, centers: np.ndarray) -> np.ndarray:
    if len(values) > 1:
        kde = gaussian_kde(values)
        return kde(centers)
    return np.zeros_like(centers)


def _conditional_mask(thicknesses_km: np.ndarray,
                      bins: np.ndarray,
                      cbe_km: float,
                      min_samples: int = 30) -> np.ndarray:
    idx = np.searchsorted(bins, cbe_km, side="right") - 1
    idx = int(np.clip(idx, 0, len(bins) - 2))
    low = bins[idx]
    high = bins[idx + 1]
    mask = (thicknesses_km >= low) & (thicknesses_km < high)
    if mask.sum() >= min_samples:
        return mask

    width = bins[1] - bins[0]
    mask = np.abs(thicknesses_km - cbe_km) <= width
    if mask.sum() >= min_samples:
        return mask

    return np.abs(thicknesses_km - cbe_km) <= 2 * width


def _mode_and_sigma(values: np.ndarray, bins: int = 30) -> tuple[float, float, float]:
    if len(values) > 1:
        x_grid = np.linspace(values.min(), values.max(), 300)
        smoothed_dense = _get_kde_pdf(values, x_grid)
        mode = float(x_grid[np.argmax(smoothed_dense)])
    else:
        counts, edges = np.histogram(values, bins=bins, density=True)
        centers = (edges[:-1] + edges[1:]) / 2
        mode = float(centers[np.argmax(counts)]) if len(counts) > 0 else 0.0
    p16 = float(np.percentile(values, 15.87))
    p84 = float(np.percentile(values, 84.13))
    return mode, mode - p16, p84 - mode


def _compute_layers(total_km: np.ndarray,
                    t_surf: np.ndarray,
                    q_v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    zc_km = np.array([
        IceConvection.lid_thickness(h_km * 1000.0, T_surface=t, T_melt=Thermal.MELT_TEMP, Q_v=q) / 1000.0
        for h_km, t, q in zip(total_km, t_surf, q_v)
    ])
    conv_km = np.maximum(total_km - zc_km, 0.0)
    return zc_km, conv_km


def main() -> None:
    data = np.load(os.path.join(RESULTS_DIR, "monte_carlo_results.npz"))
    total_km = data["thicknesses_km"]
    cbe_km = float(data["cbe_km"])
    bins = data["histogram_bins"]
    t_surf = data["param_T_surf"]
    q_v = data["param_Q_v"]

    conductive_km, convective_km = _compute_layers(total_km, t_surf, q_v)
    mask = _conditional_mask(total_km, bins, cbe_km)

    cond = conductive_km[mask]
    conv = convective_km[mask]

    bins_layers = max(20, int(np.sqrt(len(cond))))
    cond_counts, cond_edges = np.histogram(cond, bins=bins_layers, density=True)
    conv_counts, conv_edges = np.histogram(conv, bins=bins_layers, density=True)
    cond_centers = (cond_edges[:-1] + cond_edges[1:]) / 2
    conv_centers = (conv_edges[:-1] + conv_edges[1:]) / 2

    cond_pdf = _get_kde_pdf(cond, cond_centers)
    conv_pdf = _get_kde_pdf(conv, conv_centers)

    cond_mode, cond_low, cond_high = _mode_and_sigma(cond, bins=bins_layers)
    conv_mode, conv_low, conv_high = _mode_and_sigma(conv, bins=bins_layers)

    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    conductive_color = "#17BECF"
    convective_color = "#C44E52"

    ax.plot(cond_centers, cond_pdf, color=conductive_color, lw=2.0)
    ax.plot(conv_centers, conv_pdf, color=convective_color, lw=2.0)

    ax.set_xlabel("Layer Thickness [km]")
    ax.set_ylabel("Discrete Probability Density")
    x_max = max(60.0, float(np.max([cond.max() if len(cond) else 0.0, conv.max() if len(conv) else 0.0])))
    ax.set_xlim(0, x_max)

    cond_ymax = cond_pdf.max() if len(cond_pdf) else 0.0
    conv_ymax = conv_pdf.max() if len(conv_pdf) else 0.0

    cond_peak_x = cond_centers[np.argmax(cond_pdf)] if len(cond_centers) else 0.0
    conv_peak_x = conv_centers[np.argmax(conv_pdf)] if len(conv_centers) else 0.0

    ax.annotate(
        f"{cond_mode:.1f} +{cond_high:.1f} -{cond_low:.1f} km",
        xy=(cond_peak_x, cond_ymax),
        xytext=(0, 10),
        textcoords="offset points",
        color=conductive_color,
        fontsize=8,
        ha="center",
        va="bottom",
    )
    ax.annotate(
        f"{conv_mode:.1f} +{conv_high:.1f} -{conv_low:.1f} km",
        xy=(conv_peak_x, conv_ymax),
        xytext=(0, 10),
        textcoords="offset points",
        color=convective_color,
        fontsize=8,
        ha="center",
        va="bottom",
    )

    ax.text(
        cond_peak_x,
        cond_ymax * 0.55,
        "Conductive",
        color=conductive_color,
        fontsize=9,
        rotation=-70,
        ha="left",
        va="center",
    )
    ax.text(
        conv_peak_x,
        conv_ymax * 0.55,
        "Convective",
        color=convective_color,
        fontsize=9,
        rotation=-70,
        ha="left",
        va="center",
    )

    ax.text(
        0.58,
        0.25,
        f"{cbe_km:.1f} km",
        transform=ax.transAxes,
        color=conductive_color,
        fontsize=8,
        ha="left",
    )

    ax.grid(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "howell_figure4b_like.png"), dpi=300)


if __name__ == "__main__":
    main()
