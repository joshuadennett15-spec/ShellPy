"""Fixed shell parameter sets used across tests: one convecting at equilibrium, and one
conductive with a single equilibrium (high heat flux, thin shell)."""

BASE = {'D0v': 9.1e-4, 'D0b': 8.4e-4, 'd_del': 7.5e-10}
CONVECTING = {**BASE, 'd_grain': 3.413e-4, 'epsilon_0': 4.341e-6, 'mu_ice': 3.581e9, 'T_surf': 100.9,
              'D_H2O': 122.1e3, 'Q_v': 60.61e3, 'Q_b': 51.06e3, 'f_porosity': 0.0461,
              'q_radiogenic': 5.663e-3, 'q_tidal_silicate': 3.449e-3}
CONDUCTIVE = {**BASE, 'd_grain': 1.847e-3, 'epsilon_0': 2.057e-5, 'mu_ice': 2.903e9, 'T_surf': 99.25,
              'D_H2O': 107.4e3, 'Q_v': 57.45e3, 'Q_b': 50.47e3, 'f_porosity': 0.0685,
              'q_radiogenic': 8.404e-3, 'q_tidal_silicate': 2.939e-2}
