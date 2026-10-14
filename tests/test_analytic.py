"""The solver against the exact steady state of a conducting spherical shell with k = a/T.

With no internal heating, r^2 k dT/dr is constant, which for k = a/T integrates to
a ln T(r) = C / r + const. The equilibrium thickness is where the conducted flux at the
base, C / r_b^2, equals the ocean heat flux.
"""
import numpy as np
import pytest
from scipy.optimize import brentq

from shellpy import constants as c
from shellpy import physics
from shellpy.monte_carlo import run_column

from cases import CONDUCTIVE

A = 612.0   # k = A / T
P = {**CONDUCTIVE, 'epsilon_0': 0.0, 'f_porosity': 0.0}   # no tidal heating, no porosity
Q_OCEAN = 0.04


def exact_profile(H, r):
    r_b, T_m = c.RADIUS - H, physics.melting_point(H)
    C = A * np.log(T_m / P['T_surf']) / (1 / r_b - 1 / c.RADIUS)
    return C / r_b**2, np.exp(np.log(T_m) + C / A * (1 / r - 1 / r_b))


def test_conductive_equilibrium_matches_exact_solution():
    H_exact = brentq(lambda H: exact_profile(H, c.RADIUS)[0] - Q_OCEAN, 1e3, 100e3)
    s = run_column(P, Q_OCEAN, 10e3)
    assert s.converged and not s.state.is_convecting
    assert s.H == pytest.approx(H_exact, rel=1e-3)
    _, T_exact = exact_profile(s.H, c.RADIUS - s.xi * s.H)
    assert np.max(np.abs(s.T - T_exact)) < 0.01   # K
