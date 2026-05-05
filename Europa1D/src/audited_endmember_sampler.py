"""
Endmember proxy sampler for equator/mid-latitude/pole 1D MC runs.

Inherits all audited global priors from AuditedShellSampler.
Overrides only endmember-specific terms:
  T_surf:    configurable normal distribution per endmember
  epsilon_0: configurable lognormal distribution per endmember
  P_tidal:   scaled by q_tidal_multiplier (local flux adjustment)

Does NOT override d_grain — grain size is sampled once per realization
from the audited global prior (PARAMETER_PRIOR_AUDIT_2026.md).

Multiplier values (derived from 2D ocean_heat_flux shape functions at
q_star=0.45):
  Uniform:            eq=1.00, mid=1.00, pole=1.00
  Soderlund 2014:     eq=1.15, mid=0.93, pole=0.70  (equator-enhanced)
  Lemasquerier 2023:  eq=0.85, mid=1.07, pole=1.30  (polar-enhanced)

Mid-latitude (45°) parameters derived from 2D LatitudeProfile:
  T_surf = 99.1 K  (cos^1.25 interpolation, Ashkenazy 2019)
  epsilon_0 = 9.49e-6  (Beuthe 2013 eccentricity-tide pattern)
"""
import numpy as np
from typing import Dict, Tuple

from audited_sampler import AuditedShellSampler


class AuditedEndmemberSampler(AuditedShellSampler):
    """
    Endmember proxy sampler with configurable T_surf, epsilon_0, and
    local tidal flux multiplier.

    Scales only the tidal component of q_basal (P_tidal), not the
    radiogenic component, to avoid implicitly enhancing spatially
    uniform radiogenic heating.
    """

    def __init__(
        self,
        T_surf_mean: float,
        T_surf_std: float,
        T_surf_clip: Tuple[float, float],
        epsilon_0_log_center: float,
        epsilon_0_log_sigma: float,
        epsilon_0_clip: Tuple[float, float],
        q_tidal_multiplier: float = 1.0,
        seed=None,
    ):
        super().__init__(seed=seed)
        self._T_surf_mean = T_surf_mean
        self._T_surf_std = T_surf_std
        self._T_surf_clip = T_surf_clip
        self._eps_log_center = epsilon_0_log_center
        self._eps_log_sigma = epsilon_0_log_sigma
        self._eps_clip = epsilon_0_clip
        self._q_tidal_multiplier = q_tidal_multiplier

    def sample(self) -> Dict[str, float]:
        params = super().sample()

        # Override T_surf with endmember distribution
        while True:
            t = self.rng.normal(self._T_surf_mean, self._T_surf_std)
            if self._T_surf_clip[0] <= t <= self._T_surf_clip[1]:
                break
        params['T_surf'] = t

        # Override epsilon_0 with endmember distribution
        while True:
            eps = 10 ** self.rng.normal(self._eps_log_center, self._eps_log_sigma)
            if self._eps_clip[0] <= eps <= self._eps_clip[1]:
                break
        params['epsilon_0'] = eps

        # Scale tidal component of q_basal
        params['P_tidal'] = self._q_tidal_multiplier * params['P_tidal']

        # Store diagnostic
        params['q_tidal_multiplier'] = self._q_tidal_multiplier

        return params


# ── Endmember presets ────────────────────────────────────────────────────────

_EQ_PRESET = dict(
    T_surf_mean=110.0, T_surf_std=5.0, T_surf_clip=(95.0, 120.0),
    epsilon_0_log_center=np.log10(6e-6), epsilon_0_log_sigma=0.2,
    epsilon_0_clip=(2e-6, 2e-5),
)

_MID_PRESET = dict(
    T_surf_mean=99.0, T_surf_std=5.0, T_surf_clip=(85.0, 110.0),
    epsilon_0_log_center=np.log10(9.49e-6), epsilon_0_log_sigma=0.2,
    epsilon_0_clip=(2e-6, 3.0e-5),
)

_POLE_PRESET = dict(
    T_surf_mean=50.0, T_surf_std=5.0, T_surf_clip=(45.0, 80.0),
    epsilon_0_log_center=np.log10(1.2e-5), epsilon_0_log_sigma=0.2,
    epsilon_0_clip=(2e-6, 3.4e-5),
)

# q_tidal_multiplier values from 2D ocean_heat_flux() shape functions
# at q_star=0.45 (mantle_tidal_fraction=0.5):
#   Uniform:            eq=1.00, mid=1.00, pole=1.00
#   Soderlund 2014:     eq=1.15, mid=0.93, pole=0.70  (equator-enhanced)
#   Lemasquerier 2023:  eq=0.85, mid=1.07, pole=1.30  (polar-enhanced)


# ── Named subclasses (picklable for Windows multiprocessing) ─────────────────

class UniformEqSampler(AuditedEndmemberSampler):
    def __init__(self, seed=None):
        super().__init__(**_EQ_PRESET, q_tidal_multiplier=1.00, seed=seed)


class UniformPoleSampler(AuditedEndmemberSampler):
    def __init__(self, seed=None):
        super().__init__(**_POLE_PRESET, q_tidal_multiplier=1.00, seed=seed)


class SoderlundEqSampler(AuditedEndmemberSampler):
    def __init__(self, seed=None):
        super().__init__(**_EQ_PRESET, q_tidal_multiplier=1.15, seed=seed)


class SoderlundPoleSampler(AuditedEndmemberSampler):
    def __init__(self, seed=None):
        super().__init__(**_POLE_PRESET, q_tidal_multiplier=0.70, seed=seed)


class LemasquerierEqSampler(AuditedEndmemberSampler):
    def __init__(self, seed=None):
        super().__init__(**_EQ_PRESET, q_tidal_multiplier=0.85, seed=seed)


class UniformMidSampler(AuditedEndmemberSampler):
    def __init__(self, seed=None):
        super().__init__(**_MID_PRESET, q_tidal_multiplier=1.00, seed=seed)


class SoderlundMidSampler(AuditedEndmemberSampler):
    def __init__(self, seed=None):
        super().__init__(**_MID_PRESET, q_tidal_multiplier=0.93, seed=seed)


class LemasquerierMidSampler(AuditedEndmemberSampler):
    def __init__(self, seed=None):
        super().__init__(**_MID_PRESET, q_tidal_multiplier=1.07, seed=seed)


class LemasquerierPoleSampler(AuditedEndmemberSampler):
    def __init__(self, seed=None):
        super().__init__(**_POLE_PRESET, q_tidal_multiplier=1.30, seed=seed)
