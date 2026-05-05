"""
Regional Parameter Samplers for Europa Ice Shell

Custom samplers for equator vs pole comparison studies.
These capture the latitudinal variations in:
- Surface temperature
- Tidal strain amplitude
- Basal heat flux

Based on:
- Ojakangas & Stevenson (1989): Latitudinal temperature variations
- Tobie et al. (2003): Tidal dissipation patterns
- Soderlund et al. (2014): Ocean heat transport, equator-enhanced (1.15/0.70 factors)

References for parameter values provided in docstrings.
"""

import numpy as np
from typing import Dict, Optional, Tuple

from constants import Thermal, Planetary


# =============================================================================
# BASE CLASS: Shared sampling logic for all regional samplers
# =============================================================================

class _BaseRegionalSampler:
    """
    Base class containing shared parameter sampling logic.

    Subclasses override _get_regional_params() to provide region-specific
    values for T_surf, epsilon_0, and P_tidal.
    """

    def __init__(self, seed: Optional[int] = None):
        """Initialize with optional random seed."""
        self.rng = np.random.default_rng(seed)

    def _sample_truncated_normal(self, mean: float, sigma: float,
                                 low: float = -np.inf, high: float = np.inf) -> float:
        """Sample from truncated normal distribution."""
        while True:
            sample = self.rng.normal(mean, sigma)
            if low <= sample <= high:
                return sample

    def _get_regional_params(self) -> Tuple[float, float, float]:
        """
        Return region-specific (T_surf, epsilon_0, P_tidal).
        Must be overridden by subclasses.
        """
        raise NotImplementedError

    def _sample_shared_params(self, T_surf: float) -> Dict[str, float]:
        """Sample all parameters that are identical across regions."""
        d_grain = 10 ** self.rng.normal(np.log10(7e-4), 0.5)
        d_grain = np.clip(d_grain, 1e-5, 5e-3)

        D_H2O = self.rng.normal(127e3, 21e3)
        D_H2O = np.clip(D_H2O, 80e3, 200e3)

        mu_ice = self._sample_truncated_normal(3.5e9, 0.5e9, low=2.0e9, high=5.0e9)

        Q_v = self.rng.normal(59.4e3, 0.05 * 59.4e3)
        Q_b = self.rng.normal(49.0e3, 0.05 * 49.0e3)

        H_rad = self.rng.normal(4.5e-12, 1.0e-12)
        T_phi = self.rng.normal(150.0, 20.0 / 3.0)
        T_phi = np.clip(T_phi, T_surf + 1.0, Thermal.MELT_TEMP - 1.0)

        f_porosity = self.rng.uniform(0.0, 0.30)
        f_salt = 10 ** self.rng.normal(np.log10(0.03), 1.0 / 3.0)
        f_salt = np.clip(f_salt, 0.0, 0.5)
        B_k = 10 ** self.rng.uniform(-1.0, 1.0)

        D0v = max(self.rng.normal(9.1e-4, 0.033 * 9.1e-4), 1e-8)
        D0b = max(self.rng.normal(8.4e-4, 0.033 * 8.4e-4), 1e-8)
        d_del_mean = np.mean([9.04e-10, 5.22e-10])
        d_del_std = np.std([9.04e-10, 5.22e-10])
        d_del = max(self.rng.normal(d_del_mean, d_del_std), 1e-12)

        return {
            'd_grain': d_grain,
            'd_del': d_del,
            'D0v': D0v,
            'D0b': D0b,
            'mu_ice': mu_ice,
            'D_H2O': D_H2O,
            'Q_v': Q_v,
            'Q_b': Q_b,
            'H_rad': H_rad,
            'f_porosity': f_porosity,
            'f_salt': f_salt,
            'T_phi': T_phi,
            'B_k': B_k,
        }

    def sample(self) -> Dict[str, float]:
        """Sample a complete parameter set with region-specific overrides."""
        T_surf, epsilon_0, P_tidal = self._get_regional_params()

        params = self._sample_shared_params(T_surf)
        params['T_surf'] = T_surf
        params['epsilon_0'] = epsilon_0
        params['P_tidal'] = P_tidal
        return params


# =============================================================================
# DEFAULT REGIONAL SAMPLERS
# =============================================================================

class EquatorParameterSampler(_BaseRegionalSampler):
    """
    Parameter sampler for EQUATORIAL conditions on Europa.

    Characteristics:
    - Warmer surface temperature (~106-110 K)
    - Lower tidal strain amplitude
    - Lower basal heat flux (~10-15 mW/m²)

    Expected outcome: Thicker ice shell, larger conductive lid.
    """

    def _get_regional_params(self) -> Tuple[float, float, float]:
        T_surf = self.rng.normal(108.0, 2.0)
        T_surf = np.clip(T_surf, 100.0, 115.0)

        epsilon_0 = 10 ** self.rng.normal(np.log10(6e-6), 0.2)
        epsilon_0 = np.clip(epsilon_0, 1e-7, 2e-5)

        # Log-normal around 50 GW (reduced from global 100 GW)
        mean_log = np.log(50e9)
        sigma_log = np.log(10) / 4
        P_tidal = self.rng.lognormal(mean=mean_log, sigma=sigma_log)
        P_tidal = np.clip(P_tidal, 10e9, 200e9)
        return T_surf, epsilon_0, P_tidal

    def sample(self) -> Dict[str, float]:
        """Sample a complete parameter set with equator-specific overrides."""
        T_surf, epsilon_0, P_tidal = self._get_regional_params()
        params = self._sample_shared_params(T_surf)
        
        # Override d_grain for Equator to favor smaller grains (0.1 - 1 mm).
        # This dramatically lowers viscosity, ensuring convection can approach
        # the surface and cap D_cond between 10-20 km for thick shells.
        d_grain = 10 ** self.rng.normal(np.log10(3e-4), 0.3)
        params['d_grain'] = np.clip(d_grain, 1e-5, 2e-3)
        
        params['T_surf'] = T_surf
        params['epsilon_0'] = epsilon_0
        params['P_tidal'] = P_tidal
        return params


class PoleParameterSampler(_BaseRegionalSampler):
    """
    Parameter sampler for POLAR conditions on Europa.

    Characteristics:
    - Colder surface temperature (~50 K)
    - Higher tidal strain amplitude
    - Higher basal heat flux (~60-70 mW/m²) due to ocean heat transport

    Expected outcome: Thinner ice shell, more vigorous convection.
    """

    def _get_regional_params(self) -> Tuple[float, float, float]:
        T_surf = self.rng.normal(50.0, 5.0)
        T_surf = np.clip(T_surf, 35.0, 70.0)

        epsilon_0 = 10 ** self.rng.normal(np.log10(1.2e-5), 0.2)
        epsilon_0 = np.clip(epsilon_0, 5e-6, 5e-5)

        # Log-normal around 500 GW (polar-enhanced ocean heat transport)
        # Constrained to global silicate budget: total ~100-1000 GW
        mean_log = np.log(500e9)
        sigma_log = np.log(10) / 4
        P_tidal = self.rng.lognormal(mean=mean_log, sigma=sigma_log)
        P_tidal = np.clip(P_tidal, 100e9, 2000e9)

        return T_surf, epsilon_0, P_tidal


# =============================================================================
# SODERLUND ET AL. 2013 CONFIGURATION
# Ocean circulation brings MORE heat to EQUATOR, LESS to poles
# (Opposite of default configuration)
# =============================================================================

# Soderlund et al. (2014) Fig. 2/3 redistribution factors: equator receives 1.15x
# the global-mean ocean heat flux; poles receive 0.70x. Ratio = 1.64.
# Central global-mean silicate tidal power anchored at 100 GW (Tobie et al. 2003,
# Ojakangas & Stevenson 1989 Europa-scale estimates).
_SODERLUND_P_TIDAL_GLOBAL_MEAN = 100e9  # W, Europa silicate tidal budget central
_SODERLUND_EQ_FACTOR = 1.15
_SODERLUND_POLE_FACTOR = 0.70


class SoderlundEquatorSampler(_BaseRegionalSampler):
    """
    Parameter sampler for EQUATORIAL conditions - Soderlund et al. (2014).

    Equator-enhanced ocean heat transport: 1.15x the global-mean silicate
    tidal budget (Soderlund et al. 2014 Fig. 2/3).

    - Warmer surface temperature (~106-110 K)
    - Lower tidal strain amplitude
    - Ocean flux enhanced relative to global mean

    Expected outcome: THINNER ice shell at equator due to enhanced ocean heating.

    Reference: Soderlund et al. (2014), Nature Geoscience 7, 16-19,
    DOI: 10.1038/ngeo2021
    """

    def _get_regional_params(self) -> Tuple[float, float, float]:
        T_surf = self.rng.normal(108.0, 2.0)
        T_surf = np.clip(T_surf, 100.0, 115.0)

        epsilon_0 = 10 ** self.rng.normal(np.log10(6e-6), 0.2)
        epsilon_0 = np.clip(epsilon_0, 1e-7, 2e-5)

        mean_log = np.log(_SODERLUND_EQ_FACTOR * _SODERLUND_P_TIDAL_GLOBAL_MEAN)
        sigma_log = np.log(10) / 4
        P_tidal = self.rng.lognormal(mean=mean_log, sigma=sigma_log)
        P_tidal = np.clip(P_tidal, 30e9, 400e9)

        return T_surf, epsilon_0, P_tidal


class SoderlundPoleSampler(_BaseRegionalSampler):
    """
    Parameter sampler for POLAR conditions - Soderlund et al. (2014).

    Polar-suppressed ocean heat transport: 0.70x the global-mean silicate
    tidal budget (Soderlund et al. 2014 Fig. 2/3). Paired with
    SoderlundEquatorSampler to give an eq/pole ratio of 1.64.

    - Colder surface temperature (~50 K)
    - Higher tidal strain amplitude
    - Ocean flux suppressed relative to global mean

    Expected outcome: THICKER ice shell at poles due to reduced ocean heating.

    Reference: Soderlund et al. (2014), Nature Geoscience 7, 16-19,
    DOI: 10.1038/ngeo2021
    """

    def _get_regional_params(self) -> Tuple[float, float, float]:
        T_surf = self.rng.normal(50.0, 5.0)
        T_surf = np.clip(T_surf, 35.0, 70.0)

        epsilon_0 = 10 ** self.rng.normal(np.log10(1.2e-5), 0.2)
        epsilon_0 = np.clip(epsilon_0, 5e-6, 5e-5)

        mean_log = np.log(_SODERLUND_POLE_FACTOR * _SODERLUND_P_TIDAL_GLOBAL_MEAN)
        sigma_log = np.log(10) / 4
        P_tidal = self.rng.lognormal(mean=mean_log, sigma=sigma_log)
        P_tidal = np.clip(P_tidal, 20e9, 250e9)

        return T_surf, epsilon_0, P_tidal


# Quick test
if __name__ == "__main__":
    print("=== EQUATOR SAMPLER TEST ===")
    eq_sampler = EquatorParameterSampler(seed=42)
    eq_params = eq_sampler.sample()
    print(f"  T_surf:    {eq_params['T_surf']:.1f} K")
    print(f"  epsilon_0: {eq_params['epsilon_0']:.2e}")
    print(f"  P_tidal:   {eq_params['P_tidal']/1e9:.1f} GW")

    print("\n=== POLE SAMPLER TEST ===")
    pole_sampler = PoleParameterSampler(seed=42)
    pole_params = pole_sampler.sample()
    print(f"  T_surf:    {pole_params['T_surf']:.1f} K")
    print(f"  epsilon_0: {pole_params['epsilon_0']:.2e}")
    print(f"  P_tidal:   {pole_params['P_tidal']/1e9:.1f} GW")

    print("\n=== SODERLUND EQUATOR SAMPLER TEST ===")
    sod_eq = SoderlundEquatorSampler(seed=42)
    sod_eq_params = sod_eq.sample()
    print(f"  T_surf:    {sod_eq_params['T_surf']:.1f} K")
    print(f"  epsilon_0: {sod_eq_params['epsilon_0']:.2e}")
    print(f"  P_tidal:   {sod_eq_params['P_tidal']/1e9:.1f} GW")

    print("\n=== SODERLUND POLE SAMPLER TEST ===")
    sod_pole = SoderlundPoleSampler(seed=42)
    sod_pole_params = sod_pole.sample()
    print(f"  T_surf:    {sod_pole_params['T_surf']:.1f} K")
    print(f"  epsilon_0: {sod_pole_params['epsilon_0']:.2e}")
    print(f"  P_tidal:   {sod_pole_params['P_tidal']/1e9:.1f} GW")
