"""Fast parameter search for 27-35km peak."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from runtime_support import configure_numeric_runtime, default_worker_count

configure_numeric_runtime()

import numpy as np

from Monte_Carlo import MonteCarloRunner, SolverConfig

class TestSampler:
    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)
        
    def _sample_truncated_normal(self, mean, sigma, low=-np.inf, high=np.inf):
        while True:
            sample = self.rng.normal(mean, sigma)
            if low <= sample <= high: return sample
            
    def _sample_lognormal(self, mean, sigma_orders):
        sigma = sigma_orders * np.log(10)
        return self.rng.lognormal(np.log(mean), sigma)
        
    def sample(self) -> dict:
        d_grain = 10 ** self.rng.normal(np.log10(2.5e-4), 0.4) # 0.25 mm avg
        d_grain = np.clip(d_grain, 1e-5, 5e-3)

        epsilon_0 = 10 ** self.rng.normal(np.log10(1.5e-5), 0.3)
        
        T_surf = self.rng.normal(104.0, 7.0)
        D_H2O = self.rng.normal(127e3, 21e3)
        mu_ice = self._sample_truncated_normal(3.5e9, 0.5e9, low=3.5e9 / 20, high=3.5e9)

        Q_v = self.rng.normal(59.4e3, 0.05 * 59.4e3)
        Q_b = self.rng.normal(49.0e3, 0.05 * 49.0e3)
        H_rad = self.rng.normal(4.5e-12, 1.0e-12)
        T_phi = self.rng.normal(150.0, 20.0 / 3.0)

        mean_log = np.log(250e9) # 250 GW 
        P_tidal = self.rng.lognormal(mean=mean_log, sigma=np.log(10)/3)

        f_porosity = self.rng.uniform(0.0, 0.30)
        f_salt = self._sample_lognormal(0.03, 1.0)
        f_salt = np.clip(f_salt, 0.0, 0.5)
        B_k = 10 ** self.rng.uniform(-1.0, 1.0)

        D0v = self.rng.normal(9.1e-4, 0.033 * 9.1e-4)
        D0b = self.rng.normal(8.4e-4, 0.033 * 8.4e-4)
        d_del = self.rng.normal(7.13e-10, 1.9e-10)

        return {
            'd_grain': d_grain, 'epsilon_0': epsilon_0, 'T_surf': T_surf, 'D_H2O': D_H2O,
            'mu_ice': mu_ice, 'Q_v': Q_v, 'Q_b': Q_b, 'H_rad': H_rad, 'T_phi': T_phi,
            'P_tidal': P_tidal, 'f_porosity': f_porosity, 'f_salt': f_salt, 'B_k': B_k,
            'D0v': D0v, 'D0b': D0b, 'd_del': d_del
        }

if __name__ == "__main__":
    runner = MonteCarloRunner(n_iterations=200, seed=42, n_workers=default_worker_count(),
                              sampler_class=TestSampler, verbose=False)
    results = runner.run()
    
    print(f"\nVALID SAMPLES: {results.n_valid}/200")
    print(f"CBE (mode): {results.cbe_km:.1f} km")
    print(f"Median:     {np.median(results.thicknesses_km):.1f} km")
    print(f"Mean:       {np.mean(results.thicknesses_km):.1f} km")
