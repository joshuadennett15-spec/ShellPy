"""
Constrained mid-latitude sampler for iterative Juno MWR inversion.

Reads prior ranges from a JSON configuration file, allowing iterative
tightening of priors between Monte Carlo runs.

The JSON file is written by the driver script (run_midlat_juno_refit.py)
before each MC run, and read fresh by each worker process at
instantiation time (compatible with Windows spawn multiprocessing).

Parameters controlled via JSON:
  q_basal range:   Q_BASAL_LO, Q_BASAL_HI  (overrides AuditedShellSampler)
  d_grain prior:   D_GRAIN_LOG_CENTER, D_GRAIN_LOG_SIGMA, D_GRAIN_LO/HI
  T_surf:          mean, std, clip  (endmember-level override)
  epsilon_0:       log center, sigma, clip  (endmember-level override)
  q_tidal_mult:    local ocean flux multiplier  (endmember-level override)
"""
import json
import os

import numpy as np

from audited_endmember_sampler import AuditedEndmemberSampler

_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), '..', 'results', 'midlat_juno', 'current_priors.json'
)


class JunoConstrainedMidLatSampler(AuditedEndmemberSampler):
    """
    Mid-latitude endmember sampler with Juno-constrained priors.

    Reads configuration from current_priors.json at instantiation,
    enabling iterative prior tightening between MC campaigns.
    """

    def __init__(self, seed=None):
        with open(_CONFIG_PATH) as f:
            p = json.load(f)

        super().__init__(
            T_surf_mean=p['T_surf_mean'],
            T_surf_std=p['T_surf_std'],
            T_surf_clip=tuple(p['T_surf_clip']),
            epsilon_0_log_center=p['eps_log_center'],
            epsilon_0_log_sigma=p['eps_log_sigma'],
            epsilon_0_clip=tuple(p['eps_clip']),
            q_tidal_multiplier=p['q_tidal_mult'],
            seed=seed,
        )

        # Override AuditedShellSampler class-level prior ranges.
        # These instance attributes shadow the class attributes when
        # sample() accesses self.Q_BASAL_LO etc.
        self.Q_BASAL_LO = p['q_basal_lo']
        self.Q_BASAL_HI = p['q_basal_hi']
        self.D_GRAIN_LOG_CENTER = p['d_grain_log_center']
        self.D_GRAIN_LOG_SIGMA = p['d_grain_log_sigma']
        self.D_GRAIN_LO = p['d_grain_lo']
        self.D_GRAIN_HI = p['d_grain_hi']
