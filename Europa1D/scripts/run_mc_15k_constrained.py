"""
15,000-iteration Monte Carlo with physically constrained parameter distributions.

Changes from default Howell sampler:
  P_tidal:  lognormal(100 GW) truncated to [50 GW, 1 TW]
            - Lower bound: ocean maintenance (Hussmann et al., 2002)
            - Upper bound: Laplace resonance stability (Hussmann & Spohn, 2004)
  d_grain:  lognormal(0.7 mm) with lower bound raised from 10 um to 100 um
            - Grain growth over geological time (Barr & McKinnon, 2007)
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from runtime_support import configure_numeric_runtime

configure_numeric_runtime()

import numpy as np
import multiprocessing as mp

from Monte_Carlo import (
    MonteCarloRunner, SolverConfig, HowellParameterSampler, save_results,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')


class ConstrainedHowellSampler(HowellParameterSampler):
    """
    Howell (2021) distributions with physically motivated truncation.

    P_tidal: lognormal(100 GW, sigma=log10/3) clipped to [50 GW, 1 TW]
      - 50 GW lower bound: minimum silicate tidal power to maintain
        Europa's present-day ocean (Hussmann et al., 2002)
      - 1 TW upper bound: Laplace resonance orbital stability limit
        (Hussmann & Spohn, 2004)

    d_grain: lognormal(0.7 mm, 0.5 orders) clipped to [0.1 mm, 5 mm]
      - 0.1 mm lower bound: grain growth in warm convecting ice over
        geological timescales (Barr & McKinnon, 2007)
    """

    def sample(self):
        params = super().sample()

        # Re-draw P_tidal with physical truncation
        # Keep drawing until within bounds (rejection sampling)
        mean_log = np.log(100e9)
        sigma_log = np.log(10) / 3
        while True:
            P = self.rng.lognormal(mean=mean_log, sigma=sigma_log)
            if 50e9 <= P <= 1000e9:
                break
        params['P_tidal'] = P

        # Tighten d_grain lower bound
        d = params['d_grain']
        if d < 1e-4:  # 0.1 mm
            # Re-draw with tighter clip
            while True:
                d = 10 ** self.rng.normal(np.log10(7e-4), 0.5)
                if 1e-4 <= d <= 5e-3:
                    break
            params['d_grain'] = d

        return params


def main():
    config = SolverConfig(reject_subcritical=False)

    print("=" * 60)
    print("CONSTRAINED MONTE CARLO")
    print("  P_tidal: lognormal(100 GW) truncated to [50 GW, 1 TW]")
    print("  d_grain: lognormal(0.7 mm) truncated to [0.1 mm, 5 mm]")
    print("  All other params: Howell (2021) Table 1")
    print("  15,000 iterations")
    print("=" * 60)

    runner = MonteCarloRunner(
        n_iterations=15000,
        seed=42,
        verbose=True,
        config=config,
        sampler_class=ConstrainedHowellSampler,
    )
    results = runner.run()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    save_results(results, os.path.join(RESULTS_DIR, "mc_15000_constrained.npz"))

    print()
    print("=" * 60)
    print("COMPARISON WITH LITERATURE")
    print("=" * 60)
    print(f"This work CBE:       {results.cbe_km:.1f} km")
    print(f"This work median:    {results.median_km:.1f} km")
    print(f"This work 1-sigma:   [{results.sigma_1_low_km:.1f}, {results.sigma_1_high_km:.1f}] km")
    print(f"Howell (2021) CBE:   ~24.3 km")
    print(f"Juno MWR:            29 +/- 10 km (Levin et al., 2026)")
    print(f"Tobie et al. (2002): ~30 km")


if __name__ == "__main__":
    mp.freeze_support()
    main()
