"""
Parameter sampler for 2D Monte Carlo runs.

Shared shell physics is drawn from the audited 1D baseline so the 2D model is
directly comparable to the 1D workflow in the sibling ``Europa1D/src``
package. Latitude structure is then imposed separately through
`LatitudeProfile`.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'Europa1D', 'src'))

import numpy as np
from typing import Dict, Optional, Tuple

from audited_sampler import AuditedShellSampler
from constants import Planetary
from latitude_profile import LatitudeProfile, OceanPattern
from literature_scenarios import SURFACE_PRESETS, SurfacePreset


class LatitudeParameterSampler:
    """
    Sample shared shell physics plus latitude-structure controls for Europa2D.

    Shared shell properties are held constant across latitude columns within one
    realization:
        d_grain, mu_ice, Q_v, Q_b, H_rad, D_H2O, f_porosity,
        f_salt, B_k, T_phi, D0v, D0b, d_del.

    Latitude dependence enters only through:
        T_eq/T_floor, epsilon_eq/epsilon_pole, q_ocean_mean via ocean pattern,
        q_star, and mantle_tidal_fraction.
    """

    SHARED_AUDITED_KEYS = (
        'd_grain', 'd_del', 'D0v', 'D0b',
        'mu_ice', 'D_H2O', 'Q_v', 'Q_b',
        'H_rad', 'f_porosity', 'f_salt', 'T_phi', 'B_k',
    )

    LATITUDE_STRUCTURE_KEYS = (
        'T_eq', 'T_floor', 'epsilon_eq', 'epsilon_pole',
        'q_ocean_mean', 'ocean_pattern', 'ocean_amplitude',
        'q_star', 'mantle_tidal_fraction', 'tidal_pattern',
    )

    # No implicit tidal uplift in the default path.  Pass q_tidal_scale > 1
    # explicitly when running sensitivity experiments (see vetting note §E).
    TIDAL_FLUX_SCALE = 1.0

    def __init__(
        self,
        seed: Optional[int] = None,
        ocean_pattern: OceanPattern = "uniform",
        ocean_amplitude: Optional[float] = None,
        q_star: Optional[float] = None,
        tidal_pattern: str = "mantle_core",
        q_tidal_scale: float = TIDAL_FLUX_SCALE,
        grain_latitude_mode: str = "global",
        grain_strain_exponent: float = 0.5,
        surface_preset: str = "ashkenazy_low_q",
        grain_center_mm: float = 0.6,
    ):
        # Surface BC from named preset — no more hardcoded magic numbers
        if surface_preset not in SURFACE_PRESETS:
            raise ValueError(
                f"Unknown surface_preset={surface_preset!r}. "
                f"Valid: {list(SURFACE_PRESETS.keys())}"
            )
        preset = SURFACE_PRESETS[surface_preset]
        self.surface_preset = surface_preset
        self.T_floor_mean = float(preset.T_floor)

        seed_sequence = np.random.SeedSequence(seed)
        shared_seq, latitude_seq = seed_sequence.spawn(2)

        self._shared_sampler = AuditedShellSampler()
        self._shared_sampler.rng = np.random.default_rng(shared_seq)

        # Override grain prior center so it's an explicit campaign choice
        self._shared_sampler.D_GRAIN_LOG_CENTER = np.log10(grain_center_mm * 1e-3)
        self.grain_center_mm = float(grain_center_mm)

        self.rng = np.random.default_rng(latitude_seq)

        self.ocean_pattern = ocean_pattern
        self.ocean_amplitude = ocean_amplitude
        self.q_star_override = q_star
        self._tidal_pattern = tidal_pattern
        if q_tidal_scale <= 0.0:
            raise ValueError("q_tidal_scale must be positive.")
        self.q_tidal_scale = float(q_tidal_scale)
        self._grain_latitude_mode = grain_latitude_mode
        self._grain_strain_exponent = float(grain_strain_exponent)

    @classmethod
    def shared_parameter_names(cls) -> Tuple[str, ...]:
        """Shell properties that do not vary by latitude in this 2D proxy."""
        return cls.SHARED_AUDITED_KEYS

    @classmethod
    def latitude_structure_names(cls) -> Tuple[str, ...]:
        """Controls that define the latitude dependence of one realization."""
        return cls.LATITUDE_STRUCTURE_KEYS

    def sample(self) -> Tuple[Dict[str, float], LatitudeProfile]:
        """
        Sample all parameters for one 2D MC iteration.

        Returns:
            (shared_params, latitude_profile) tuple
        """
        audited_params = self._shared_sampler.sample()
        D_H2O = audited_params['D_H2O']
        H_rad = audited_params['H_rad']

        # Reuse the audited 1D surface-temperature draw at the equator so the
        # 2D equatorial anchor matches the 1D prior exactly for each realization.
        T_eq = float(np.clip(audited_params['T_surf'], 80.0, 120.0))

        # Sample a warmer polar floor than the old Ashkenazy-only default while
        # preserving a modest spread around the requested mean.
        floor_low = max(1.0, self.T_floor_mean - 8.0)
        floor_high = self.T_floor_mean + 9.0
        T_floor = self.rng.normal(self.T_floor_mean, 4.0)
        T_floor = float(np.clip(T_floor, floor_low, floor_high))
        T_floor = min(T_floor, T_eq - 1.0)

        # Latitude-varying tidal strain uses equatorial and polar anchors.
        # Tightened: 0.15 dex sigma (was 0.2); clips match 1D proxy priors.
        epsilon_eq = 10 ** self.rng.normal(np.log10(6e-6), 0.15)
        epsilon_eq = np.clip(epsilon_eq, 3e-6, 1.2e-5)

        epsilon_pole = 10 ** self.rng.normal(np.log10(1.2e-5), 0.15)
        epsilon_pole = np.clip(epsilon_pole, 6e-6, 2.5e-5)

        # Mean basal heat flux inherits the audited 1D shell prior. In the
        # current 2D proxy that global-mean basal flux is redistributed by the
        # latitude-only ocean pattern. Apply only a modest 2D-only uplift to
        # the tidal/ocean component so the 1D audited prior itself is not
        # rewritten.
        R_rock = Planetary.RADIUS - D_H2O
        M_rock = (4.0 / 3.0) * np.pi * (R_rock ** 3) * 3500.0
        q_radiogenic = (H_rad * M_rock) / Planetary.AREA
        q_tidal_inherited = audited_params['P_tidal'] / Planetary.AREA
        q_tidal_global = self.q_tidal_scale * q_tidal_inherited
        q_basal_global = q_radiogenic + q_tidal_global
        q_ocean_mean = q_basal_global

        mantle_tidal_fraction = float(self.rng.uniform(0.1, 0.9))

        # For equator-enhanced cases, keep a direct q* prior matched to the
        # Soderlund-style benchmark. Polar/uniform cases derive q* from the
        # tidal fraction unless a scenario override is provided.
        q_star_explicit = self.q_star_override
        if q_star_explicit is None and self.ocean_pattern == "equator_enhanced":
            q_star_explicit = self.rng.normal(0.4, 0.1)
            q_star_explicit = float(np.clip(q_star_explicit, 0.1, 0.8))

        # tidal_pattern and grain mode are held fixed per MC campaign, not sampled
        tidal_pattern = self._tidal_pattern
        grain_latitude_mode = self._grain_latitude_mode

        profile = LatitudeProfile(
            T_eq=T_eq,
            T_floor=T_floor,
            epsilon_eq=epsilon_eq,
            epsilon_pole=epsilon_pole,
            q_ocean_mean=q_ocean_mean,
            ocean_pattern=self.ocean_pattern,
            ocean_amplitude=self.ocean_amplitude,
            q_star=q_star_explicit,
            mantle_tidal_fraction=mantle_tidal_fraction,
            tidal_pattern=tidal_pattern,
            grain_latitude_mode=grain_latitude_mode,
            grain_strain_exponent=self._grain_strain_exponent,
        )

        shared_params = {
            key: audited_params[key]
            for key in self.SHARED_AUDITED_KEYS
        }
        shared_params['q_basal'] = q_basal_global
        shared_params['q_tidal'] = q_tidal_global
        shared_params['q_basal_inherited'] = q_radiogenic + q_tidal_inherited
        shared_params['q_tidal_inherited'] = q_tidal_inherited
        shared_params['q_tidal_scale'] = self.q_tidal_scale

        return shared_params, profile
