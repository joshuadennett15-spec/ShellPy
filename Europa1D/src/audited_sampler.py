"""
2026-audited parameter sampler for shell-level thickness study (Option A).

Primary changes from Howell (2021):
  q_basal:   decomposed prior — q_rad ~ TruncNorm(7, 1, [4, 12]) mW/m^2
             + q_tidal ~ LogUniform(2, 20) mW/m^2.  Gives right-skewed
             total with median ~13, mean ~15 mW/m^2.
             References: Barr & Showman (2009), Behounkova et al. (2021),
             Lemasquerier et al. (2023).
  d_grain:   lognormal(0.6 mm, 0.35 dex), clipped [0.05, 4.0] mm
             (EUROPA_GRAIN_SIZE_PRIOR_REVIEW_2026.md; Mitri 2023, Ruiz et al. 2007)
  f_salt:    fixed at 0 (pure-ice baseline; PARAMETER_PRIOR_AUDIT_2026.md)
  B_k:       fixed at 1
  epsilon_0: clip tightened to [2e-6, 3.4e-5]
  f_porosity: narrowed to [0, 0.10]
  T_surf:    N(104, 7) clipped [80, 120] K (Howell 2021)
  T_phi:     fixed at 150.0 K (PARAMETER_PRIOR_AUDIT_2026.md)
  H_rad:     truncated positive
"""
import numpy as np

from Monte_Carlo import HowellParameterSampler
from constants import Planetary


class AuditedShellSampler(HowellParameterSampler):
    """
    Howell (2021) framework with 2026 audit revisions.

    q_basal is sampled as two independent physical components:

      q_radiogenic ~ TruncNorm(7.0, 1.0, [4, 12]) mW/m^2
        Chondritic mantle heating.  Center from Barr & Showman (2009);
        width reflects compositional uncertainty (CI vs LL chondrite).

      q_tidal_silicate ~ LogUniform(2, 20) mW/m^2
        Silicate mantle tidal dissipation.  Spans Behounkova et al. (2021)
        present-day minimum (~60 GW) to aggressive estimates (~600 GW).
        Log-uniform reflects order-of-magnitude uncertainty.

      q_basal = q_radiogenic + q_tidal_silicate
        Effective range [6, 32] mW/m^2; median ~13, mean ~15.

    d_grain: lognormal(0.6 mm, 0.35 dex), clipped [0.05, 4.0] mm.
      Centers on Mitri (2023) self-consistent tidal-reduction results
      (0.39-0.80 mm). Encompasses Ruiz et al. (2007) Europa-specific
      convective range (0.2-2 mm) and McKinnon (1999) ~1 mm scale.
      Reference: EUROPA_GRAIN_SIZE_PRIOR_REVIEW_2026.md.
    """

    # Radiogenic component: TruncNorm(mu, sigma, [lo, hi]) in W/m^2
    Q_RAD_MU = 0.007        # 7.0 mW/m^2  (Barr & Showman 2009)
    Q_RAD_SIGMA = 0.001     # 1.0 mW/m^2
    Q_RAD_LO = 0.004        # 4.0 mW/m^2
    Q_RAD_HI = 0.012        # 12.0 mW/m^2

    # Silicate tidal component: LogUniform(lo, hi) in W/m^2
    Q_TIDAL_LOG_LO = 0.002  # 2.0 mW/m^2  (~60 GW, Behounkova+ 2021 present-day min)
    Q_TIDAL_LOG_HI = 0.020  # 20.0 mW/m^2 (~600 GW)

    # Optional total-q_basal rejection bounds (W/m^2).
    # Default: permissive (no rejection).  Subclasses such as
    # JunoConstrainedMidLatSampler may tighten these for iterative
    # prior narrowing.
    Q_BASAL_LO = 0.0        # 0 mW/m^2
    Q_BASAL_HI = 0.050      # 50 mW/m^2

    # Grain size prior: lognormal centered at 0.6 mm, sigma 0.35 dex
    D_GRAIN_LOG_CENTER = np.log10(6e-4)   # log10(0.6 mm in meters)
    D_GRAIN_LOG_SIGMA = 0.35
    D_GRAIN_LO = 5e-5    # 0.05 mm
    D_GRAIN_HI = 4e-3    # 4.0 mm

    def _sample_q_radiogenic(self):
        """Sample mantle radiogenic flux (W/m^2) from truncated normal."""
        while True:
            q = self.rng.normal(self.Q_RAD_MU, self.Q_RAD_SIGMA)
            if self.Q_RAD_LO <= q <= self.Q_RAD_HI:
                return q

    def _sample_q_tidal_silicate(self):
        """Sample silicate tidal flux (W/m^2) from log-uniform."""
        log_lo = np.log10(self.Q_TIDAL_LOG_LO)
        log_hi = np.log10(self.Q_TIDAL_LOG_HI)
        return 10 ** self.rng.uniform(log_lo, log_hi)

    def sample(self):
        params = super().sample()

        # 1. Sample decomposed basal flux, with optional rejection bounds
        while True:
            q_rad = self._sample_q_radiogenic()
            q_tidal = self._sample_q_tidal_silicate()
            q_basal = q_rad + q_tidal
            if self.Q_BASAL_LO <= q_basal <= self.Q_BASAL_HI:
                break
        params['P_tidal'] = q_tidal * Planetary.AREA

        # 2. Literature-constrained grain size (Mitri 2023, Ruiz et al. 2007)
        while True:
            d = 10 ** self.rng.normal(self.D_GRAIN_LOG_CENTER, self.D_GRAIN_LOG_SIGMA)
            if self.D_GRAIN_LO <= d <= self.D_GRAIN_HI:
                break
        params['d_grain'] = d

        # 3. Pure-ice baseline
        params['f_salt'] = 0.0
        params['B_k'] = 1.0

        # 4. Tighten epsilon_0 clip
        eps = params['epsilon_0']
        if eps < 2e-6 or eps > 3.4e-5:
            while True:
                eps = 10 ** self.rng.normal(np.log10(1.2e-5), 0.3)
                if 2e-6 <= eps <= 3.4e-5:
                    break
            params['epsilon_0'] = eps

        # 5. Fix T_phi (PARAMETER_PRIOR_AUDIT_2026.md: not worth broadening)
        params['T_phi'] = 150.0

        # 6. Narrow f_porosity
        params['f_porosity'] = self.rng.uniform(0.0, 0.10)

        # 7. Tighten T_surf clip (Howell 2021: N(104, 7))
        T = params['T_surf']
        if T < 80.0 or T > 120.0:
            params['T_surf'] = np.clip(
                self.rng.normal(104.0, 7.0), 80.0, 120.0
            )

        # 8. Truncate H_rad positive
        if params['H_rad'] <= 0:
            params['H_rad'] = abs(self.rng.normal(4.5e-12, 1.0e-12))

        return params
