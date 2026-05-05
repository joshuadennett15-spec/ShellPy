"""
Fixed-seed regression tests for 1D convection model.

These tests anchor the numerical output of the model at known parameter
combinations.  If a code change shifts any of these values, the test
fails — forcing the developer to decide whether the shift is intentional
(update the anchor) or a bug.

All cases use NU_SCALING="dv2021" with FK viscosity.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pytest
import tempfile

from Solver import Thermal_Solver
from Boundary_Conditions import FixedTemperature
from constants import Convection as ConvConst


T_SURF = 96.0   # Ashkenazy low-Q baseline
PARAMS = {
    'd_grain': 1e-3, 'Q_v': 59.4e3, 'Q_b': 49.0e3,
    'mu_ice': 3.3e9, 'D0v': 9.1e-4, 'D0b': 8.4e-4,
    'd_del': 7.13e-10, 'f_porosity': 0.0, 'f_salt': 0.0,
    'B_k': 1.0, 'T_phi': 150.0, 'epsilon_0': 1e-5,
    'T_surf': T_SURF,
}


class TestRegressionSubcritical:
    """Subcritical (conductive) regime: thin shell, high ocean flux."""

    def _run(self):
        bc = FixedTemperature(temperature=T_SURF)
        solver = Thermal_Solver(
            nx=31, thickness=15e3, dt=1e11,
            surface_bc=bc, use_convection=True,
            physics_params=PARAMS,
        )
        for _ in range(500):
            v = solver.solve_step(0.025)
            if abs(v) < 1e-12:
                break
        return solver

    def test_thickness(self):
        solver = self._run()
        assert solver.H / 1000.0 == pytest.approx(16.773, abs=0.01)

    def test_D_cond(self):
        solver = self._run()
        assert solver.convection_state.D_cond / 1000.0 == pytest.approx(14.815, abs=0.01)

    def test_D_conv(self):
        solver = self._run()
        assert solver.convection_state.D_conv / 1000.0 == pytest.approx(1.954, abs=0.01)

    def test_Ra_subcritical(self):
        solver = self._run()
        assert solver.convection_state.Ra == pytest.approx(143.16, rel=0.01)
        assert not solver.convection_state.is_convecting

    def test_nu_scaling_recorded(self):
        solver = self._run()
        assert solver.convection_state.nu_scaling == "dv2021"


class TestRegressionSupercritical:
    """Supercritical (convecting) regime: thick shell, low ocean flux."""

    def _run(self):
        bc = FixedTemperature(temperature=T_SURF)
        solver = Thermal_Solver(
            nx=31, thickness=30e3, dt=1e12,
            surface_bc=bc, use_convection=True,
            physics_params=PARAMS,
        )
        for _ in range(1500):
            v = solver.solve_step(0.012)
            if abs(v) < 1e-12:
                break
        return solver

    def test_thickness(self):
        solver = self._run()
        assert solver.H / 1000.0 == pytest.approx(30.000, abs=0.05)

    def test_D_cond(self):
        solver = self._run()
        assert solver.convection_state.D_cond / 1000.0 == pytest.approx(23.765, abs=0.05)

    def test_D_conv(self):
        solver = self._run()
        assert solver.convection_state.D_conv / 1000.0 == pytest.approx(6.240, abs=0.05)

    def test_Ra_supercritical(self):
        solver = self._run()
        assert solver.convection_state.Ra == pytest.approx(4588.3, rel=0.01)
        assert solver.convection_state.is_convecting

    def test_Nu(self):
        solver = self._run()
        assert solver.convection_state.Nu == pytest.approx(2.747, rel=0.01)


class TestNpzRoundTrip:
    """Saved .npz files must preserve closure and BC metadata."""

    def test_metadata_round_trip(self):
        from Monte_Carlo import MonteCarloResults, save_results, load_results

        dummy = MonteCarloResults(
            thicknesses_km=np.array([20.0, 25.0]),
            n_iterations=2,
            n_valid=2,
            histogram_bins=np.array([15.0, 20.0, 25.0, 30.0]),
            histogram_counts=np.array([0, 1, 1]),
            pdf_smoothed=np.array([0.0, 0.5, 0.5]),
            bin_centers=np.array([17.5, 22.5, 27.5]),
            cbe_km=22.5,
            median_km=22.5,
            mean_km=22.5,
            sigma_1_low_km=20.0,
            sigma_1_high_km=25.0,
            runtime_seconds=1.0,
            nu_scaling="green",
            run_metadata={
                "nu_scaling": "green",
                "ra_crit": "1000.0",
                "sampler_class": "HowellParameterSampler",
                "seed": "42",
            },
        )
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            save_results(dummy, path)
            loaded = load_results(path)
            assert loaded.nu_scaling == "green"
            assert loaded.run_metadata is not None
            assert loaded.run_metadata["nu_scaling"] == "green"
            assert loaded.run_metadata["ra_crit"] == "1000.0"
            assert loaded.run_metadata["sampler_class"] == "HowellParameterSampler"
            assert loaded.run_metadata["seed"] == "42"
        finally:
            os.unlink(path)
