"""
Equatorial-proxy sampler for Juno MWR comparison.

Inherits all audited global priors from AuditedShellSampler.
Overrides only equatorial-specific terms:
  T_surf:    N(96, 4) clipped [90, 104] K — annual-mean equatorial
             (Ashkenazy 2019; Levin et al. 2026)
  epsilon_0: lognormal(6e-6, 0.15 dex) clipped [3e-6, 1.2e-5] (Tobie+ 2003)
  q_tidal:   scaled by enhancement_factor (ocean heat redistribution)

Does NOT override d_grain — grain size is sampled once per realization
from the audited global prior (PARAMETER_PRIOR_AUDIT_2026.md).

Enhancement modes:
  0.55x — strong equatorial depletion (Lemasquerier+ 2023, q*=0.91)
  0.67x — conservative equatorial depletion (Lemasquerier+ 2023, 2:1 pole/eq)
  1.0x  — uniform ocean transport (Ashkenazy+ 2021)
  1.15x — equatorial enhancement (Soderlund+ 2014, Fig. 2/3)
  1.5x  — strong equatorial enhancement (upper-bound sensitivity)
"""
import numpy as np

from audited_sampler import AuditedShellSampler
from constants import Planetary


class AuditedEquatorialSampler(AuditedShellSampler):
    """
    Equatorial-proxy sampler with ocean heat enhancement modes.

    Scales only the tidal/ocean component of q_basal, not the
    radiogenic component, to avoid implicitly enhancing spatially
    uniform radiogenic heating.
    """

    def __init__(self, seed=None, enhancement_factor=1.0):
        super().__init__(seed=seed)
        self.enhancement_factor = enhancement_factor

    def sample(self):
        params = super().sample()

        # 1. Override T_surf: equatorial annual-mean
        #    N(96, 4) clipped [90, 104] K (Ashkenazy 2019; Levin+ 2026)
        while True:
            t = self.rng.normal(96.0, 4.0)
            if 90.0 <= t <= 104.0:
                break
        params['T_surf'] = t

        # 2. Override epsilon_0: equatorial tidal strain is lower
        #    lognormal(6e-6, 0.15 dex) clipped [3e-6, 1.2e-5]
        while True:
            eps = 10 ** self.rng.normal(np.log10(6e-6), 0.15)
            if 3e-6 <= eps <= 1.2e-5:
                break
        params['epsilon_0'] = eps

        # 3. Scale tidal component of q_basal (not radiogenic)
        #    Parent already set P_tidal = q_silicate_tidal * AREA.
        #    We read it back, scale, and rewrite. The solver reads P_tidal
        #    to compute q_basal = q_radiogenic + P_tidal / AREA.
        if self.enhancement_factor != 1.0:
            q_tidal_flux = params['P_tidal'] / Planetary.AREA
            params['P_tidal'] = self.enhancement_factor * q_tidal_flux * Planetary.AREA

        # 4. Store diagnostics
        params['eq_enhancement'] = self.enhancement_factor

        return params
