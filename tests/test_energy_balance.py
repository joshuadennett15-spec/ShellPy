"""At equilibrium, heat lost through the surface must equal ocean heat flux plus tidal heating."""
import numpy as np
import pytest
from scipy.integrate import trapezoid

from shellpy.convection import face_conductivity
from shellpy.monte_carlo import basal_heat_flux
from shellpy.solver import ThermalSolver

from cases import CONDUCTIVE, CONVECTING


def surface_to_supplied_ratio(p):
    q_ocean = basal_heat_flux(p)
    s = ThermalSolver(p, 25e3, spherical=False)
    assert s.run(q_ocean), "did not reach equilibrium"
    k, _, _, q_tidal, state = s.properties(s.T)
    dz = s.H / (s.nx - 1)
    q_surface = face_conductivity(k)[0] * (s.T[1] - s.T[0]) / dz
    return q_surface / (q_ocean + trapezoid(q_tidal, dx=dz)), state


def test_conductive_shell_conserves_energy():
    ratio, state = surface_to_supplied_ratio(CONDUCTIVE)
    assert not state.is_convecting
    assert ratio == pytest.approx(1.0, abs=0.005)


def test_convecting_shell_conserves_energy():
    ratio, state = surface_to_supplied_ratio(CONVECTING)
    assert state.is_convecting
    assert ratio == pytest.approx(1.0, abs=0.005)
