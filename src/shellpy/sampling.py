"""Prior distributions of the shell parameters: Howell (2021) with the 2026 audit revisions."""
import numpy as np


def _truncated(draw, lo, hi):
    """Redraw until the value lies in [lo, hi]."""
    while True:
        x = draw()
        if lo <= x <= hi:
            return float(x)


def sample_parameters(rng):
    """One draw of every shell parameter, as a dict (SI units).

    q_radiogenic and q_tidal_silicate are the heat fluxes the rocky interior delivers
    to the base of the ice through the ocean; their sum is the basal heat flux.
    """
    return {
        # grain size: Mitri (2023), Ruiz et al. (2007); 0.6 mm, 0.35 dex
        'd_grain': _truncated(lambda: 10 ** rng.normal(np.log10(6e-4), 0.35), 5e-5, 4e-3),
        'd_del': _truncated(lambda: rng.normal(7.13e-10, 1.91e-10), 1e-12, np.inf),   # grain boundary width (m)
        'D0v': rng.normal(9.1e-4, 0.033 * 9.1e-4),          # volume diffusion prefactor (m^2/s)
        'D0b': rng.normal(8.4e-4, 0.033 * 8.4e-4),          # boundary diffusion prefactor (m^2/s)
        'Q_v': rng.normal(59.4e3, 0.05 * 59.4e3),           # volume diffusion activation energy (J/mol)
        'Q_b': rng.normal(49.0e3, 0.05 * 49.0e3),           # boundary diffusion activation energy (J/mol)
        'epsilon_0': _truncated(lambda: 10 ** rng.normal(np.log10(1.2e-5), 0.3), 2e-6, 3.4e-5),   # tidal strain
        'mu_ice': _truncated(lambda: rng.normal(3.5e9, 0.5e9), 2.0e9, 5.0e9),                    # shear modulus (Pa)
        'T_surf': _truncated(lambda: rng.normal(104.0, 7.0), 80.0, 120.0),                       # K
        'D_H2O': _truncated(lambda: rng.normal(127e3, 21e3), 80e3, 200e3),                       # hydrosphere depth (m)
        'f_porosity': rng.uniform(0.0, 0.10),               # porosity of ice colder than 150 K
        # basal heat flux (W/m^2): Barr & Showman (2009); Behounkova et al. (2021)
        'q_radiogenic': _truncated(lambda: rng.normal(7e-3, 1e-3), 4e-3, 12e-3),
        'q_tidal_silicate': 10 ** rng.uniform(np.log10(2e-3), np.log10(20e-3)),
    }
