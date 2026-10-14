"""Prior distributions for the shell parameters (Howell 2021 with the 2026 audit revisions)."""
import numpy as np

from shellpy import constants as c

PARAMETERS = ('d_grain', 'd_del', 'D0v', 'D0b', 'epsilon_0', 'mu_ice', 'T_surf', 'D_H2O',
              'Q_v', 'Q_b', 'H_rad', 'P_tidal', 'f_porosity', 'f_salt', 'T_phi', 'B_k')


def _truncated(draw, lo, hi):
    while True:
        x = draw()
        if lo <= x <= hi:
            return x


def sample_parameters(rng):
    """One draw of every shell parameter, as a dict keyed by PARAMETERS (SI units).

    Draw order matches ShellPy v1.0.1 so a given seed reproduces the thesis ensembles;
    several Howell (2021) draws are immediately replaced by the audited priors below.
    """
    p = {}
    p['d_grain'] = 10 ** rng.normal(np.log10(7e-4), 0.5)          # replaced below
    eps = 10 ** rng.normal(np.log10(1.2e-5), 0.3)
    T_surf = rng.normal(104.0, 7.0)
    p['D_H2O'] = float(np.clip(rng.normal(127e3, 21e3), 80e3, 200e3))
    p['mu_ice'] = _truncated(lambda: rng.normal(3.5e9, 0.5e9), 2.0e9, 5.0e9)
    p['Q_v'] = rng.normal(59.4e3, 0.05 * 59.4e3)
    p['Q_b'] = rng.normal(49.0e3, 0.05 * 49.0e3)
    p['H_rad'] = rng.normal(4.5e-12, 1.0e-12)                       # W/kg
    rng.normal(150.0, 20.0 / 3.0)                                   # old T_phi draw, replaced below
    rng.lognormal(np.log(100e9), np.log(10) / 3)                    # old P_tidal draw, replaced below
    rng.uniform(0.0, 0.30)                                          # old porosity draw, replaced below
    rng.normal(np.log10(0.03), 1.0 / 3.0)                           # old salt draw, replaced below
    rng.uniform(-1.0, 1.0)                                          # old B_k draw, replaced below
    p['D0v'] = max(rng.normal(9.1e-4, 0.033 * 9.1e-4), 1e-8)        # m^2/s
    p['D0b'] = max(rng.normal(8.4e-4, 0.033 * 8.4e-4), 1e-8)        # m^2/s
    p['d_del'] = max(rng.normal(np.mean([9.04e-10, 5.22e-10]), np.std([9.04e-10, 5.22e-10])), 1e-12)
    T_surf = float(np.clip(T_surf, 50.0, 150.0))
    eps = float(np.clip(eps, 1e-7, 1e-3))

    # 2026 audit revisions
    _truncated(lambda: rng.normal(0.007, 0.001), 0.004, 0.012)      # q_rad (W/m^2): drawn but unused, see notes
    p['P_tidal'] = 10 ** rng.uniform(np.log10(0.002), np.log10(0.020)) * c.AREA   # W
    p['d_grain'] = _truncated(lambda: 10 ** rng.normal(np.log10(6e-4), 0.35), 5e-5, 4e-3)   # m
    p['f_salt'] = 0.0
    p['B_k'] = 1.0
    if not 2e-6 <= eps <= 3.4e-5:
        eps = _truncated(lambda: 10 ** rng.normal(np.log10(1.2e-5), 0.3), 2e-6, 3.4e-5)
    p['epsilon_0'] = eps
    p['T_phi'] = 150.0
    p['f_porosity'] = rng.uniform(0.0, 0.10)
    p['T_surf'] = T_surf if 80.0 <= T_surf <= 120.0 else float(np.clip(rng.normal(104.0, 7.0), 80.0, 120.0))
    if p['H_rad'] <= 0:
        p['H_rad'] = abs(rng.normal(4.5e-12, 1.0e-12))
    return {k: float(p[k]) for k in PARAMETERS}
