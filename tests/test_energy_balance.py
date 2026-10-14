"""At equilibrium, heat lost through the surface must equal ocean heat flux plus tidal heating."""
import pytest
from scipy.integrate import trapezoid

from shellpy import constants as c
from shellpy.convection import face_conductivity
from shellpy.monte_carlo import basal_heat_flux, run_column

from cases import CONDUCTIVE, CONVECTING


def surface_to_supplied_ratio(p):
    """Power lost at the surface over power supplied by the ocean and tides (per steradian)."""
    q_ocean = basal_heat_flux(p)
    s = run_column(p, q_ocean, 25e3)
    assert s.converged
    k, _, _, q_tidal, state = s.properties(s.T)
    dz = s.H / (s.nx - 1)
    r = c.RADIUS - s.xi * s.H
    lost = face_conductivity(k)[0] * (s.T[1] - s.T[0]) / dz * ((r[0] + r[1]) / 2) ** 2
    supplied = q_ocean * r[-1] ** 2 + trapezoid(q_tidal * r**2, dx=dz)
    return lost / supplied, state


def test_conductive_shell_conserves_energy():
    ratio, state = surface_to_supplied_ratio(CONDUCTIVE)
    assert not state.is_convecting
    assert ratio == pytest.approx(1.0, abs=0.005)


def test_convecting_shell_conserves_energy():
    ratio, state = surface_to_supplied_ratio(CONVECTING)
    assert state.is_convecting
    assert ratio == pytest.approx(1.0, abs=0.005)
