"""
Polar-proxy sampler for comparison with equatorial-proxy MC runs.

Inherits all audited global priors from AuditedShellSampler.
Overrides only polar-specific terms:
  T_surf:    N(46, 4) clipped [38, 55] K — annual-mean polar
             (Ashkenazy 2019; Levin et al. 2026)
  epsilon_0: lognormal(1.2e-5, 0.15 dex) clipped [6e-6, 2.5e-5]
             (Tobie+ 2003)

Does NOT override d_grain or P_tidal --- grain size is sampled from the
audited global prior, and basal heat flux is left at the global value so
that this run isolates the effect of polar surface conditions only.

References:
  Ashkenazy (2019), Levin et al. (2026): Polar annual-mean T_surf
  Tobie et al. (2003): Polar tidal strain ~2x equatorial
"""
import numpy as np

from audited_sampler import AuditedShellSampler


class AuditedPolarSampler(AuditedShellSampler):
    """
    Polar-proxy sampler: colder surface, higher tidal strain.

    Isolates polar surface conditions (T_surf, epsilon_0) against the
    same audited global priors used by the equatorial suite.  No ocean
    heat transport scaling is applied.
    """

    def __init__(self, seed=None):
        super().__init__(seed=seed)

    def sample(self):
        params = super().sample()

        # 1. Override T_surf: polar annual-mean
        #    N(46, 4) clipped [38, 55] K (Ashkenazy 2019; Levin+ 2026)
        while True:
            t = self.rng.normal(46.0, 4.0)
            if 38.0 <= t <= 55.0:
                break
        params['T_surf'] = t

        # 2. Override epsilon_0: polar tidal strain is higher
        #    lognormal(1.2e-5, 0.15 dex) clipped [6e-6, 2.5e-5]
        #    Tobie et al. (2003): polar strain ~2x equatorial
        while True:
            eps = 10 ** self.rng.normal(np.log10(1.2e-5), 0.15)
            if 6e-6 <= eps <= 2.5e-5:
                break
        params['epsilon_0'] = eps

        return params