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
