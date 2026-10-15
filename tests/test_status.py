"""Every sample is labelled with what physically happened, and keeps its outputs."""
import numpy as np

from shellpy.monte_carlo import solve_1d

from cases import BISTABLE, CONVECTING


def test_shell_thicker_than_the_hydrosphere_is_frozen_through():
    out = solve_1d({**CONVECTING, 'D_H2O': 20e3}, start='thick')
    assert out['status'] == 'frozen through'
    assert out['H_km'] >= 20.0 and np.isfinite(out['Nu'])


def test_test_cases_are_labelled():
    assert solve_1d(BISTABLE, start='thick')['status'] == 'ok'
    assert solve_1d(CONVECTING, start='thick')['status'] == 'frozen through'   # equilibrium 142 km > D_H2O
