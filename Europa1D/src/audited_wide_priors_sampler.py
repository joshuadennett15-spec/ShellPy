"""
Wide literature-envelope sensitivity sampler.

This is NOT the new baseline. The audited prior (AuditedShellSampler) remains
the internally consistent canonical specification. This subclass widens the
genuinely literature-uncertain rheology and heat-budget parameters to test
whether the weak Juno D_cond update is an artefact of prior narrowing.

Kept from audit (do not undo):
  - q_basal decomposition: q_rad ~ TruncNorm + q_tidal ~ LogUniform
  - Pure-ice baseline (f_salt=0, B_k=1, T_phi=150 K)
  - Truncate H_rad positive

Widened relative to audit:
  d_grain    : lognormal(1.0 mm, 0.5 dex), clip [0.05, 10] mm
               Barr & McKinnon (2007), De La Chapelle (1999),
               Goldsby & Kohlstedt (2001). Genuinely unconstrained;
               0.5 dex permits mm-to-cm tail without making cm grains
               the central tendency.
  epsilon_0  : lognormal(1e-5, 0.4 dex), clip [1e-6, 1e-4] s^-1
               Tobie (2003), Beuthe (2013), Han & Showman (2010).
  q_tidal    : LogUniform(1, 50) mW/m^2  (was 2-20)
               Behounkova (2021), Tobie (2003) full envelope.
  q_radiogenic: TruncNorm(7, 2) clip [3, 12] mW/m^2  (was sigma=1)
               Spohn & Schubert (2003), Hussmann & Spohn (2004).
  Q_v        : N(59.4 kJ/mol, sigma=5 kJ/mol)   (was sigma~3)
  Q_b        : N(49.0 kJ/mol, sigma=5 kJ/mol)   (was sigma~2.5)
               Goldsby & Kohlstedt (2001) experimental scatter.
  f_porosity : Uniform(0, 0.20)                 (was [0, 0.10])
               Nimmo & Manga (2009).

Unchanged (well-anchored or scope-defining):
  mu_ice, D_H2O, T_surf, H_rad, f_salt, B_k, T_phi.
"""
import numpy as np

from audited_sampler import AuditedShellSampler


class AuditedWidePriorsSampler(AuditedShellSampler):
    """Literature-envelope wide priors. Sensitivity test, not baseline."""

    # Wider basal flux components
    Q_RAD_SIGMA = 0.002     # 2 mW/m^2 (was 1)
    Q_RAD_LO = 0.003        # 3 mW/m^2 (was 4)
    Q_TIDAL_LOG_LO = 0.001  # 1 mW/m^2 (was 2)
    Q_TIDAL_LOG_HI = 0.050  # 50 mW/m^2 (was 20)

    # Wider grain size: lognormal(1.0 mm, 0.5 dex) clip [0.05, 10] mm
    D_GRAIN_LOG_CENTER = np.log10(1.0e-3)
    D_GRAIN_LOG_SIGMA = 0.5
    D_GRAIN_LO = 5e-5    # 0.05 mm
    D_GRAIN_HI = 1e-2    # 10 mm

    # Wider strain rate clip
    EPS_LOG_CENTER = np.log10(1e-5)
    EPS_LOG_SIGMA = 0.4
    EPS_LO = 1e-6
    EPS_HI = 1e-4

    # Wider activation energies (Goldsby & Kohlstedt 2001 experimental scatter)
    Q_V_MEAN = 59.4e3
    Q_V_SIGMA = 5.0e3
    Q_B_MEAN = 49.0e3
    Q_B_SIGMA = 5.0e3

    # Wider porosity (Nimmo & Manga 2009)
    F_PORO_HI = 0.20

    def sample(self):
        params = super().sample()

        # Re-draw epsilon_0 with wider envelope
        while True:
            eps = 10 ** self.rng.normal(self.EPS_LOG_CENTER, self.EPS_LOG_SIGMA)
            if self.EPS_LO <= eps <= self.EPS_HI:
                break
        params['epsilon_0'] = eps

        # Re-draw activation energies with literature-defensible widths
        params['Q_v'] = self.rng.normal(self.Q_V_MEAN, self.Q_V_SIGMA)
        params['Q_b'] = self.rng.normal(self.Q_B_MEAN, self.Q_B_SIGMA)

        # Re-draw porosity to wider envelope
        params['f_porosity'] = self.rng.uniform(0.0, self.F_PORO_HI)

        return params
