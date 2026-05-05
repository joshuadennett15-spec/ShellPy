from regional_samplers import _BaseRegionalSampler
from typing import Tuple
import numpy as np

# =============================================================================
# 500 GW BUDGET SCENARIOS (docs/ops/Run.md)
# =============================================================================

class Run1EquatorSampler(_BaseRegionalSampler):
    """Run 1: First Principles (Polar Concentrated Heating) - 500 GW Budget"""
    def _get_regional_params(self) -> Tuple[float, float, float]:
        T_surf = self.rng.normal(108.0, 2.0)
        T_surf = np.clip(T_surf, 100.0, 115.0)
        epsilon_0 = 10 ** self.rng.normal(np.log10(6e-6), 0.2)
        epsilon_0 = np.clip(epsilon_0, 1e-7, 2e-5)
        
        # 50 GW
        mean_log = np.log(50e9)
        sigma_log = np.log(10) / 4
        P_tidal = self.rng.lognormal(mean=mean_log, sigma=sigma_log)
        P_tidal = np.clip(P_tidal, 10e9, 200e9)
        return T_surf, epsilon_0, P_tidal

    def sample(self):
        T_surf, epsilon_0, P_tidal = self._get_regional_params()
        params = self._sample_shared_params(T_surf)
        d_grain = 10 ** self.rng.normal(np.log10(3e-4), 0.3) # keep small grain size preference for equator
        params['d_grain'] = np.clip(d_grain, 1e-5, 2e-3)
        params['T_surf'] = T_surf
        params['epsilon_0'] = epsilon_0
        params['P_tidal'] = P_tidal
        return params

class Run1PoleSampler(_BaseRegionalSampler):
    """Run 1: First Principles (Polar Concentrated Heating) - 500 GW Budget"""
    def _get_regional_params(self) -> Tuple[float, float, float]:
        T_surf = self.rng.normal(50.0, 5.0)
        T_surf = np.clip(T_surf, 35.0, 70.0)
        epsilon_0 = 10 ** self.rng.normal(np.log10(1.2e-5), 0.2)
        epsilon_0 = np.clip(epsilon_0, 5e-6, 5e-5)
        
        # 450 GW
        mean_log = np.log(450e9)
        sigma_log = np.log(10) / 4
        P_tidal = self.rng.lognormal(mean=mean_log, sigma=sigma_log)
        P_tidal = np.clip(P_tidal, 100e9, 1500e9)
        return T_surf, epsilon_0, P_tidal

class Run2EquatorSampler(_BaseRegionalSampler):
    """Run 2: Equatorial Heating (Soderlund Model) - 500 GW Budget"""
    def _get_regional_params(self) -> Tuple[float, float, float]:
        T_surf = self.rng.normal(108.0, 2.0)
        T_surf = np.clip(T_surf, 100.0, 115.0)
        epsilon_0 = 10 ** self.rng.normal(np.log10(6e-6), 0.2)
        epsilon_0 = np.clip(epsilon_0, 1e-7, 2e-5)
        
        # 290 GW
        mean_log = np.log(290e9)
        sigma_log = np.log(10) / 4
        P_tidal = self.rng.lognormal(mean=mean_log, sigma=sigma_log)
        P_tidal = np.clip(P_tidal, 50e9, 1000e9)
        return T_surf, epsilon_0, P_tidal

    def sample(self):
        T_surf, epsilon_0, P_tidal = self._get_regional_params()
        params = self._sample_shared_params(T_surf)
        d_grain = 10 ** self.rng.normal(np.log10(3e-4), 0.3)
        params['d_grain'] = np.clip(d_grain, 1e-5, 2e-3)
        params['T_surf'] = T_surf
        params['epsilon_0'] = epsilon_0
        params['P_tidal'] = P_tidal
        return params

class Run2PoleSampler(_BaseRegionalSampler):
    """Run 2: Equatorial Heating (Soderlund Model) - 500 GW Budget"""
    def _get_regional_params(self) -> Tuple[float, float, float]:
        T_surf = self.rng.normal(50.0, 5.0)
        T_surf = np.clip(T_surf, 35.0, 70.0)
        epsilon_0 = 10 ** self.rng.normal(np.log10(1.2e-5), 0.2)
        epsilon_0 = np.clip(epsilon_0, 5e-6, 5e-5)
        
        # 210 GW
        mean_log = np.log(210e9)
        sigma_log = np.log(10) / 4
        P_tidal = self.rng.lognormal(mean=mean_log, sigma=sigma_log)
        P_tidal = np.clip(P_tidal, 50e9, 800e9)
        return T_surf, epsilon_0, P_tidal

class Run3EquatorSampler(_BaseRegionalSampler):
    """Run 3: Uniform Distribution (Efficient Ocean Mixing) - 500 GW Budget"""
    def _get_regional_params(self) -> Tuple[float, float, float]:
        T_surf = self.rng.normal(108.0, 2.0)
        T_surf = np.clip(T_surf, 100.0, 115.0)
        epsilon_0 = 10 ** self.rng.normal(np.log10(6e-6), 0.2)
        epsilon_0 = np.clip(epsilon_0, 1e-7, 2e-5)
        
        # 250 GW
        mean_log = np.log(250e9)
        sigma_log = np.log(10) / 4
        P_tidal = self.rng.lognormal(mean=mean_log, sigma=sigma_log)
        P_tidal = np.clip(P_tidal, 50e9, 800e9)
        return T_surf, epsilon_0, P_tidal

    def sample(self):
        T_surf, epsilon_0, P_tidal = self._get_regional_params()
        params = self._sample_shared_params(T_surf)
        d_grain = 10 ** self.rng.normal(np.log10(3e-4), 0.3)
        params['d_grain'] = np.clip(d_grain, 1e-5, 2e-3)
        params['T_surf'] = T_surf
        params['epsilon_0'] = epsilon_0
        params['P_tidal'] = P_tidal
        return params

class Run3PoleSampler(_BaseRegionalSampler):
    """Run 3: Uniform Distribution (Efficient Ocean Mixing) - 500 GW Budget"""
    def _get_regional_params(self) -> Tuple[float, float, float]:
        T_surf = self.rng.normal(50.0, 5.0)
        T_surf = np.clip(T_surf, 35.0, 70.0)
        epsilon_0 = 10 ** self.rng.normal(np.log10(1.2e-5), 0.2)
        epsilon_0 = np.clip(epsilon_0, 5e-6, 5e-5)
        
        # 250 GW
        mean_log = np.log(250e9)
        sigma_log = np.log(10) / 4
        P_tidal = self.rng.lognormal(mean=mean_log, sigma=sigma_log)
        P_tidal = np.clip(P_tidal, 50e9, 800e9)
        return T_surf, epsilon_0, P_tidal
