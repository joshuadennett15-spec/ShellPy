"""Fixed shell parameter sets used across tests: one convecting so vigorously that it freezes
the ocean through (equilibrium 128 km, beyond its 122 km hydrosphere), one conductive (high heat
flux, thin shell), and one convecting at equilibrium over its ocean (48.7 km; before the DV2021
whole-shell closure it had two equilibria)."""

BASE = {'D0v': 9.1e-4, 'D0b': 8.4e-4, 'd_del': 7.5e-10}
CONVECTING = {**BASE, 'd_grain': 3.413e-4, 'epsilon_0': 4.341e-6, 'mu_ice': 3.581e9, 'T_surf': 100.9,
              'D_H2O': 122.1e3, 'Q_v': 60.61e3, 'Q_b': 51.06e3, 'f_porosity': 0.0461,
              'q_radiogenic': 5.663e-3, 'q_tidal_silicate': 3.449e-3}
CONDUCTIVE = {**BASE, 'd_grain': 1.847e-3, 'epsilon_0': 2.057e-5, 'mu_ice': 2.903e9, 'T_surf': 99.25,
              'D_H2O': 107.4e3, 'Q_v': 57.45e3, 'Q_b': 50.47e3, 'f_porosity': 0.0685,
              'q_radiogenic': 8.404e-3, 'q_tidal_silicate': 2.939e-2}
CONVECTING_OCEAN = {**BASE, 'd_grain': 4.951e-4, 'epsilon_0': 1.067e-5, 'mu_ice': 3.863e9, 'T_surf': 96.27,
            'D_H2O': 132.9e3, 'Q_v': 60.35e3, 'Q_b': 49.59e3, 'f_porosity': 0.06878,
            'q_radiogenic': 6.012e-3, 'q_tidal_silicate': 8.847e-3}
