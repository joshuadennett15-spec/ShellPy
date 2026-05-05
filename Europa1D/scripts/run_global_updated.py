"""
Global Monte Carlo run with updated, physically plausible P_tidal (150-350 GW).
Runs 15,000 iterations, saves results, and generates the standard plots.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from runtime_support import configure_numeric_runtime, default_worker_count

configure_numeric_runtime()

import numpy as np
import multiprocessing as mp

from Monte_Carlo import (
    MonteCarloRunner, SolverConfig, HowellParameterSampler, save_results
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')


class UpdatedPtidalSampler(HowellParameterSampler):
    """
    Same as default Howell sampler but with P_tidal drawn uniformly
    between 150 GW and 350 GW, which is more physically plausible
    based on modern tidal dissipation estimates for Europa's silicate mantle.
    """

    def sample(self):
        params = super().sample()
        # Override P_tidal: uniform between 150 GW and 350 GW
        params['P_tidal'] = self.rng.uniform(150e9, 350e9)
        return params


def main():
    config = SolverConfig(reject_subcritical=False)
    n_workers = default_worker_count()

    print("=" * 60)
    print("GLOBAL MONTE CARLO: P_tidal ~ U(150, 350) GW")
    print("15,000 iterations")
    print(f"Workers: {n_workers}")
    print("=" * 60)

    runner = MonteCarloRunner(
        n_iterations=15000,
        seed=42,
        verbose=True,
        n_workers=n_workers,
        config=config,
        sampler_class=UpdatedPtidalSampler,
    )
    results = runner.run()

    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    save_results(results, os.path.join(RESULTS_DIR, "global_updated_ptidal.npz"))

    # --- Generate plots ---
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde

    thickness = results.thicknesses_km

    # Plot 1: Total thickness PDF
    kde = gaussian_kde(thickness)
    x = np.linspace(max(0, thickness.min() - 5), thickness.max() + 5, 300)

    fig1, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.hist(thickness, bins=60, density=True, alpha=0.35, color="#4C72B0", label="Histogram")
    ax1.plot(x, kde(x), color="#C44E52", lw=2.0, label="KDE (Smoothed PDF)")
    ax1.axvline(results.cbe_km, color="k", ls="--", lw=1.2,
                label=f"CBE = {results.cbe_km:.1f} km")
    ax1.set_xlabel("Ice shell thickness (km)")
    ax1.set_ylabel("Probability density")
    ax1.set_title(f"Global Monte Carlo (N={results.n_valid:,}, P_tidal 150-350 GW)")
    ax1.legend()
    ax1.grid(True, alpha=0.2)
    fig1.tight_layout()
    fig1.savefig(os.path.join(FIGURES_DIR, "monte_carlo_results_updated.png"), dpi=200)
    print(f"Saved: monte_carlo_results_updated.png")

    # Plot 2: Conductive lid thickness
    d_cond = results.D_cond_km
    kde_d = gaussian_kde(d_cond)
    x_d = np.linspace(max(0, d_cond.min() - 2), d_cond.max() + 2, 300)

    fig2, ax2 = plt.subplots(figsize=(7, 4.5))
    ax2.hist(d_cond, bins=50, density=True, alpha=0.35, color="#4C72B0", label="Histogram")
    ax2.plot(x_d, kde_d(x_d), color="#C44E52", lw=2.0, label="KDE (Smoothed PDF)")
    ax2.set_xlabel("Conductive Lid Thickness ($D_{cond}$) [km]")
    ax2.set_ylabel("Probability density")
    ax2.set_title(f"Global Conductive Lid (N={results.n_valid:,}, P_tidal 150-350 GW)")
    ax2.legend()
    ax2.grid(True, alpha=0.2)
    fig2.tight_layout()
    fig2.savefig(os.path.join(FIGURES_DIR, "monte_carlo_dcond_updated.png"), dpi=200)
    print(f"Saved: monte_carlo_dcond_updated.png")

    # Also generate shell structure plot
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    from plot_shell_structure import plot_single
    plot_single("global_updated_ptidal.npz", "Global (P_tidal 150-350 GW)",
                "shell_structure_global_updated.png")


if __name__ == "__main__":
    mp.freeze_support()
    main()
