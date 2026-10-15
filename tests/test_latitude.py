"""Latitude forcing keeps its global means and anchors, and a small 2D run completes."""
import numpy as np
import pytest
from scipy.integrate import quad

from shellpy.latitude import SCENARIOS, ocean_heat_flux, sample_2d, solve_2d, surface_temperature, tidal_strain


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_ocean_heat_flux_preserves_area_weighted_mean(scenario):
    mean = quad(lambda phi: ocean_heat_flux(phi, 0.02, scenario) * np.cos(phi), 0, np.pi / 2)[0]
    assert mean == pytest.approx(0.02, rel=1e-9)


def test_forcing_hits_equator_and_pole_anchors():
    assert surface_temperature(0.0, 104.0, 46.0) == pytest.approx(104.0)
    assert surface_temperature(np.pi / 2, 104.0, 46.0) == pytest.approx(46.0)
    assert tidal_strain(np.pi / 2, 6e-6, 1.2e-5) == pytest.approx(1.2e-5)


def test_small_2d_run():
    out = solve_2d(sample_2d(4, seed=1), scenario='lemasquerier2023_polar', start='thin', n_lat=3)
    assert out['status'] == 'ok'
    assert out['H_km'].shape == (3,) and np.all(out['converged'])


def test_2d_sample_keeps_the_1d_values_as_area_means():
    from shellpy.monte_carlo import sample_1d
    p, s = sample_1d(3, seed=7), sample_2d(3, seed=7)
    area_mean = lambda f: quad(lambda phi: f(phi) * np.cos(phi), 0, np.pi / 2)[0]
    assert area_mean(lambda phi: surface_temperature(phi, s['T_eq'], s['T_floor'])) == pytest.approx(p['T_surf'], rel=1e-5)
    assert area_mean(lambda phi: tidal_strain(phi, s['epsilon_eq'], s['epsilon_pole']) ** 2) == pytest.approx(p['epsilon_0'] ** 2, rel=1e-9)
