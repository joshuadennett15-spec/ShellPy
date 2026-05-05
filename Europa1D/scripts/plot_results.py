import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')


def main() -> None:
    data = np.load(os.path.join(RESULTS_DIR, "monte_carlo_results.npz"))
    
    # --- PLOT 1: TOTAL THICKNESS ---
    thickness = data["thicknesses_km"]
    bins_th = data["histogram_bins"]
    counts_th = data["histogram_counts"]
    centers_th = data["bin_centers"]

    # Calculate KDE
    kde_th = gaussian_kde(thickness)
    x_th = np.linspace(max(0, thickness.min() - 5), thickness.max() + 5, 300)
    pdf_th_kde = kde_th(x_th)

    fig1, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.hist(thickness, bins=bins_th, density=True, alpha=0.35, color="#4C72B0", label="Histogram")
    ax1.plot(centers_th, counts_th, color="#4C72B0", lw=1.0, alpha=0.6, label="Discrete PDF")
    ax1.plot(x_th, pdf_th_kde, color="#C44E52", lw=2.0, label="KDE (Smoothed PDF)")
    ax1.set_xlabel("Ice shell thickness (km)")
    ax1.set_ylabel("Probability density")
    ax1.set_title("Monte Carlo Ice Shell Thickness")
    ax1.legend()
    ax1.grid(True, alpha=0.2)
    fig1.tight_layout()
    fig1.savefig(os.path.join(FIGURES_DIR, "monte_carlo_results.png"), dpi=200)

    # --- PLOT 2: CONDUCTIVE LID THICKNESS ---
    d_cond = data["D_cond_km"]
    bins_d = np.linspace(np.min(d_cond), np.max(d_cond), 40)
    counts_d, bin_edges = np.histogram(d_cond, bins=bins_d, density=True)
    centers_d = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Calculate KDE
    kde_d = gaussian_kde(d_cond)
    x_d = np.linspace(max(0, d_cond.min() - 5), d_cond.max() + 5, 300)
    pdf_d_kde = kde_d(x_d)

    fig2, ax2 = plt.subplots(figsize=(7, 4.5))
    ax2.hist(d_cond, bins=bins_d, density=True, alpha=0.35, color="#4C72B0", label="Histogram")
    ax2.plot(centers_d, counts_d, color="#4C72B0", lw=1.0, alpha=0.6, label="Discrete PDF")
    ax2.plot(x_d, pdf_d_kde, color="#C44E52", lw=2.0, label="KDE (Smoothed PDF)")
    ax2.set_xlabel("Conductive Lid Thickness ($D_{cond}$) [km]")
    ax2.set_ylabel("Probability density")
    ax2.set_title("Monte Carlo Conductive Lid Thickness")
    ax2.legend()
    ax2.grid(True, alpha=0.2)
    fig2.tight_layout()
    fig2.savefig(os.path.join(FIGURES_DIR, "monte_carlo_dcond.png"), dpi=200)


if __name__ == "__main__":
    main()
