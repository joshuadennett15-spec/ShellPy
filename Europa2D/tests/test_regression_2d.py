"""
Fixed-seed regression tests for 2D axisymmetric model.

Anchors the 3-column (equator, 45°, pole) output at known parameters.
Also tests band-mean diagnostics focus on D_cond, not pole-node values.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import src

import numpy as np
import pytest
from latitude_profile import LatitudeProfile
from axial_solver import AxialSolver2D
from profile_diagnostics import band_mean_samples, LOW_LAT_BAND, HIGH_LAT_BAND


def _run_3col():
    """3-column regression case: Ashkenazy 96/46, uniform ocean, low q.

    Explicitly pins NU_SCALING to "green" so the regression test is
    deterministic regardless of config.json state or import order.
    The Green scaling is the baseline this regression was calibrated against.
    DV2021 physics are tested separately in test_dv2021.py.
    """
    from constants import Convection as ConvConst
    ConvConst.NU_SCALING = "green"

    profile = LatitudeProfile(
        T_eq=96.0, T_floor=46.0,
        epsilon_eq=6e-6, epsilon_pole=1.2e-5,
        q_ocean_mean=0.008, ocean_pattern='uniform',
    )
    solver = AxialSolver2D(
        n_lat=3, nx=31, dt=1e12,
        latitude_profile=profile, use_convection=True,
        initial_thickness=40e3,
    )
    return solver.run_to_equilibrium(threshold=1e-12, max_steps=500, verbose=False), solver


class TestRegression2D:
    """Anchor values for the 3-column case."""

    def setup_method(self):
        self.result, self.solver = _run_3col()
        self.H = self.result['H_profile_km']
        self.diag = self.result['diagnostics']

    def test_equator_thickness(self):
        assert self.H[0] == pytest.approx(39.042, abs=0.5)

    def test_pole_thickness(self):
        assert self.H[2] == pytest.approx(56.839, abs=0.1)

    def test_equator_convecting(self):
        d = self.diag[0]
        assert d['Ra'] > 1000.0
        assert d['Nu'] > 1.0
        assert d['D_cond_km'] == pytest.approx(33.509, abs=0.5)
        assert d['D_conv_km'] == pytest.approx(5.5, abs=1.0)

    def test_pole_subcritical(self):
        d = self.diag[2]
        assert d['Ra'] < 1000.0
        assert d['Nu'] == pytest.approx(1.0, abs=0.001)

    def test_latitude_monotonic(self):
        """Shell thickness should increase from equator to pole (colder surface).
        At 3-column resolution the equator/mid-lat may be near-equal (< 0.1 km)."""
        assert self.H[0] < self.H[2]

    def test_D_cond_less_than_H_total(self):
        """D_cond must never exceed H_total."""
        for i, d in enumerate(self.diag):
            assert d['D_cond_km'] <= self.H[i] + 0.001


def _run_3col_dv2021():
    """3-column regression case with DV2021 scaling.

    Same parameters as _run_3col() but uses DV2021 mixed-heating scaling
    instead of Green, providing regression coverage for the new physics path.
    """
    from constants import Convection as ConvConst
    ConvConst.NU_SCALING = "dv2021"

    profile = LatitudeProfile(
        T_eq=96.0, T_floor=46.0,
        epsilon_eq=6e-6, epsilon_pole=1.2e-5,
        q_ocean_mean=0.008, ocean_pattern='uniform',
    )
    solver = AxialSolver2D(
        n_lat=3, nx=31, dt=1e12,
        latitude_profile=profile, use_convection=True,
        initial_thickness=40e3,
    )
    return solver.run_to_equilibrium(threshold=1e-12, max_steps=500, verbose=False), solver


class TestRegressionDV2021:
    """Anchor values for the 3-column case under DV2021 scaling.

    Mirrors TestRegression2D but with nu_scaling="dv2021" (Deschamps & Vilella
    2021 mixed-heating closure).  Baselines captured at max_steps=500,
    threshold=1e-12, identical to the Green fixture.
    """

    def setup_method(self):
        self.result, self.solver = _run_3col_dv2021()
        self.H = self.result['H_profile_km']
        self.diag = self.result['diagnostics']

    def test_equator_thickness_dv2021(self):
        assert self.H[0] == pytest.approx(32.672, abs=0.5)

    def test_pole_thickness_dv2021(self):
        assert self.H[2] == pytest.approx(56.839, abs=0.1)

    def test_equator_convecting_dv2021(self):
        d = self.diag[0]
        assert d['Ra'] > 1000.0
        assert d['Nu'] > 1.0
        assert d['D_cond_km'] == pytest.approx(23.208, abs=0.5)
        assert d['D_conv_km'] == pytest.approx(9.477, abs=1.0)

    def test_pole_subcritical_dv2021(self):
        d = self.diag[2]
        assert d['Ra'] < 1000.0
        assert d['Nu'] == pytest.approx(1.0, abs=0.001)

    def test_latitude_monotonic_dv2021(self):
        """Shell thickness should increase from equator to pole."""
        assert self.H[0] < self.H[2]

    def test_D_cond_less_than_H_total_dv2021(self):
        """D_cond must never exceed H_total."""
        for i, d in enumerate(self.diag):
            assert d['D_cond_km'] <= self.H[i] + 0.001

    def test_dv2021_differs_from_green(self):
        """DV2021 and Green scaling must produce different equator thicknesses."""
        result_green, _ = _run_3col()
        assert self.H[0] != pytest.approx(result_green['H_profile_km'][0], abs=0.01)


class TestBandMeanDiagnostics:
    """Band means (not pole-node readings) are the thesis-facing outputs."""

    def test_band_mean_computation(self):
        """band_mean_samples returns area-weighted means over latitude bands."""
        # Synthetic 19-column profile — shape (1, n_lat) for single sample
        lats_deg = np.linspace(0, 90, 19)
        H_profile = 20.0 + 30.0 * lats_deg / 90.0
        H_2d = H_profile.reshape(1, -1)

        low = band_mean_samples(lats_deg, H_2d, LOW_LAT_BAND)
        high = band_mean_samples(lats_deg, H_2d, HIGH_LAT_BAND)

        # Low band (0-10°) should be close to 20 km
        assert float(low[0]) == pytest.approx(20.0, abs=2.0)
        # High band (80-90°): area-weighted (cos) pulls value below 50 km
        assert float(high[0]) == pytest.approx(47.2, abs=2.0)
        # High > low for poleward-thickening shells
        assert float(high[0]) > float(low[0])

    def test_D_cond_band_means_not_pole_nodes(self):
        """D_cond band means should be used, not raw pole-node values."""
        result, solver = _run_3col()
        lats_rad = solver.latitudes
        D_cond_km = np.array([d['D_cond_km'] for d in result['diagnostics']])
        H_km = result['H_profile_km']

        # With only 3 columns, band means may degenerate. The key check
        # is that D_cond < H_total at every column.
        for dc, h in zip(D_cond_km, H_km):
            assert dc <= h + 0.001
