"""Fixed shell parameter sets used across tests (one convecting, one conductive at equilibrium)."""

BASE = {'D0v': 9.1e-4, 'D0b': 8.4e-4, 'd_del': 7.5e-10, 'f_salt': 0.0, 'B_k': 1.0, 'T_phi': 150.0}
CONVECTING = {**BASE, 'd_grain': 3.413e-4, 'epsilon_0': 4.341e-6, 'mu_ice': 3.581e9, 'T_surf': 100.9,
              'D_H2O': 122.1e3, 'Q_v': 60.61e3, 'Q_b': 51.06e3, 'H_rad': 3.97e-12, 'P_tidal': 1.056e11,
              'f_porosity': 0.0461}
CONDUCTIVE = {**BASE, 'd_grain': 1.847e-3, 'epsilon_0': 2.057e-5, 'mu_ice': 2.903e9, 'T_surf': 99.25,
              'D_H2O': 107.4e3, 'Q_v': 57.45e3, 'Q_b': 50.47e3, 'H_rad': 5.715e-12, 'P_tidal': 3.238e11,
              'f_porosity': 0.0685}
