"""A column reported as converged must be at equilibrium: running much longer must not move it."""
from shellpy.monte_carlo import basal_heat_flux, run_column

from cases import CONVECTING


def test_converged_column_stays_put():
    q = basal_heat_flux(CONVECTING)
    s = run_column(CONVECTING, q, 25e3)
    assert s.converged
    H = s.H
    for _ in range(5000):
        s.step(q)
    assert abs(s.H - H) < 10.0   # m


def test_unconverged_column_is_flagged():
    s = run_column(CONVECTING, basal_heat_flux(CONVECTING), 60e3, max_steps=500)
    assert not s.converged
