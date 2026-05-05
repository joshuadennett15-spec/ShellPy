"""
Validation: 2D model at a single latitude must match 1D model output.

Tests use the Ashkenazy (2019) low-Q baseline surface preset (96/46 K)
consistently.  Both 1D and 2D paths use the same closure (NU_SCALING),
rheology, and boundary conditions so differences reveal only
discretization/implementation bugs, not physics mismatches.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import src

import numpy as np
import pytest
from latitude_profile import LatitudeProfile
from axial_solver import AxialSolver2D
from Solver import Thermal_Solver
from Boundary_Conditions import FixedTemperature
from Physics import IcePhysics
from literature_scenarios import SURFACE_PRESETS


# ── Shared test constants (Ashkenazy low-Q baseline) ──────────────────────

PRESET = SURFACE_PRESETS["ashkenazy_low_q"]
T_SURF = PRESET.T_eq            # 96.0 K
Q_OCEAN = 0.025                 # W/m²
NX = 31
DT = 1e11
THICKNESS = 15e3                # m

PARAMS = {
    'd_grain': 1e-3, 'Q_v': 59.4e3, 'Q_b': 49.0e3,
    'mu_ice': 3.3e9, 'D0v': 9.1e-4, 'D0b': 8.4e-4,
    'd_del': 7.13e-10, 'f_porosity': 0.0, 'f_salt': 0.0,
    'B_k': 1.0, 'T_phi': 150.0, 'epsilon_0': 1e-5,
}


class TestSingleColumnParity:
    """2D single-column output must match standalone 1D solver under identical physics."""

    def _run_1d(self, use_convection: bool) -> float:
        bc = FixedTemperature(temperature=T_SURF)
        solver = Thermal_Solver(
            nx=NX, thickness=THICKNESS, dt=DT,
            surface_bc=bc, use_convection=use_convection,
            physics_params={**PARAMS, 'T_surf': T_SURF},
        )
        for _ in range(500):
            v = solver.solve_step(Q_OCEAN)
            if abs(v) < 1e-12:
                break
        return solver.H / 1000.0

    def _run_2d_single_column(self, use_convection: bool) -> float:
        profile = LatitudeProfile(
            T_eq=T_SURF,
            epsilon_eq=PARAMS['epsilon_0'],
            epsilon_pole=PARAMS['epsilon_0'],
            q_ocean_mean=Q_OCEAN,
            ocean_pattern="uniform",
            surface_pattern="uniform",  # T_eq everywhere — matches 1D fixed BC
        )
        solver = AxialSolver2D(
            n_lat=1, nx=NX, dt=DT,
            latitude_profile=profile, physics_params=PARAMS,
            use_convection=use_convection, initial_thickness=THICKNESS,
        )
        result = solver.run_to_equilibrium(
            threshold=1e-12, max_steps=500, verbose=False,
        )
        return result['H_profile_km'][0]

    def test_conductive_parity(self):
        """Conductive-only: 2D single column must match 1D within 1%."""
        H_1d = self._run_1d(use_convection=False)
        H_2d = self._run_2d_single_column(use_convection=False)
        assert H_2d == pytest.approx(H_1d, rel=0.01), \
            f"2D ({H_2d:.3f} km) vs 1D ({H_1d:.3f} km)"

    def test_convective_parity(self):
        """Convective: 2D single column must match 1D within 2%."""
        H_1d = self._run_1d(use_convection=True)
        H_2d = self._run_2d_single_column(use_convection=True)
        assert H_2d == pytest.approx(H_1d, rel=0.02), \
            f"2D ({H_2d:.3f} km) vs 1D ({H_1d:.3f} km)"


class TestSanityBounds:
    """Ashkenazy baseline and legacy sensitivity must produce physical shells."""

    def test_ashkenazy_baseline(self):
        """Ashkenazy (2019) low-Q baseline (96/46 K) single column."""
        profile = LatitudeProfile(
            T_eq=96.0, T_floor=46.0,
            epsilon_eq=6e-6, epsilon_pole=6e-6,
            q_ocean_mean=0.02, ocean_pattern="uniform",
        )
        solver = AxialSolver2D(
            n_lat=1, nx=31, dt=1e12,
            latitude_profile=profile, use_convection=True,
        )
        result = solver.run_to_equilibrium(
            threshold=1e-12, max_steps=1500, verbose=False,
        )
        H = result["H_profile_km"][0]
        assert 5.0 < H < 80.0, f"Unphysical thickness: {H:.1f} km"

    def test_legacy_sensitivity(self):
        """Legacy 110/52 K BCs — sensitivity comparison only."""
        profile = LatitudeProfile(
            T_eq=110.0, T_floor=52.0,
            epsilon_eq=6e-6, epsilon_pole=6e-6,
            q_ocean_mean=0.02, ocean_pattern="uniform",
        )
        solver = AxialSolver2D(
            n_lat=1, nx=31, dt=1e12,
            latitude_profile=profile, use_convection=True,
        )
        result = solver.run_to_equilibrium(
            threshold=1e-12, max_steps=1500, verbose=False,
        )
        H = result["H_profile_km"][0]
        assert 5.0 < H < 80.0, f"Unphysical thickness: {H:.1f} km"
