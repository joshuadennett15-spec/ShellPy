"""At equilibrium, heat lost through the surface must equal ocean heat flux plus tidal heating."""
import numpy as np
import pytest
from scipy.integrate import trapezoid

from shellpy.convection import face_conductivity
from shellpy.monte_carlo import basal_heat_flux
from shellpy.solver import ThermalSolver

BASE = {'D0v': 9.1e-4, 'D0b': 8.4e-4, 'd_del': 7.5e-10, 'f_salt': 0.0, 'B_k': 1.0, 'T_phi': 150.0}
CONVECTING = {**BASE, 'd_grain': 3.413e-4, 'epsilon_0': 4.341e-6, 'mu_ice': 3.581e9, 'T_surf': 100.9,
              'D_H2O': 122.1e3, 'Q_v': 60.61e3, 'Q_b': 51.06e3, 'H_rad': 3.97e-12, 'P_tidal': 1.056e11,
              'f_porosity': 0.0461}
CONDUCTIVE = {**BASE, 'd_grain': 1.847e-3, 'epsilon_0': 2.057e-5, 'mu_ice': 2.903e9, 'T_surf': 99.25,
              'D_H2O': 107.4e3, 'Q_v': 57.45e3, 'Q_b': 50.47e3, 'H_rad': 5.715e-12, 'P_tidal': 3.238e11,
              'f_porosity': 0.0685}


def surface_to_supplied_ratio(p):
    q_ocean = basal_heat_flux(p)
    s = ThermalSolver(p, 25e3, spherical=False)
    for _ in range(8000):
        s.step(q_ocean)
    assert abs(s.step(q_ocean)) < 1e-15, "did not reach equilibrium"
    k, _, _, q_tidal, state = s.properties(s.T)
    dz = s.H / (s.nx - 1)
    q_surface = face_conductivity(k)[0] * (s.T[1] - s.T[0]) / dz
    return q_surface / (q_ocean + trapezoid(q_tidal, dx=dz)), state


def test_conductive_shell_conserves_energy():
    ratio, state = surface_to_supplied_ratio(CONDUCTIVE)
    assert not state.is_convecting
    assert ratio == pytest.approx(1.0, abs=0.03)


@pytest.mark.xfail(strict=True, reason="known bug: the Stefan condition uses plain k, not the Nu-enhanced k")
def test_convecting_shell_conserves_energy():
    ratio, state = surface_to_supplied_ratio(CONVECTING)
    assert state.is_convecting
    assert ratio == pytest.approx(1.0, abs=0.03)
