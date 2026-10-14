"""Where a shell ends up can depend on where it starts (Mitri & Showman 2005), but only for
real reasons: the two branches are a conductive and a convecting shell."""
import pytest

from shellpy.monte_carlo import basal_heat_flux, run_column, solve_1d, warm_start

from cases import BISTABLE, CONVECTING


def test_two_equilibria_are_conductive_and_convecting():
    thin, thick = (solve_1d(BISTABLE, start=s) for s in ('thin', 'thick'))
    assert thin['status'] == thick['status'] == 'ok'
    assert thin['D_conv_km'] == 0 and thick['D_conv_km'] > 0
    assert thick['H_km'] > thin['H_km'] + 3


def test_single_equilibrium_is_reached_from_both_starts():
    q = basal_heat_flux(CONVECTING)
    thin, thick = (run_column(CONVECTING, q, warm_start(CONVECTING, q, s)) for s in ('thin', 'thick'))
    assert thin.H == pytest.approx(thick.H, abs=1.0)   # m
