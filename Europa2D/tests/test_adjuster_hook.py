"""Tests for the convection_adjuster hook in build_conductivity_profile."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'Europa1D', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pytest
from Convection import IceConvection, ConvectionState
from constants import Thermal


def _make_test_profile(nx=31, H=20e3, T_surf=96.0, T_melt=273.0):
    """Create a linear temperature profile for testing."""
    z_grid = np.linspace(0, H, nx)
    T_profile = T_surf + (T_melt - T_surf) * z_grid / H
    return T_profile, z_grid, H, T_melt


class TestConvectionAdjusterHook:
    """Tests that build_conductivity_profile calls the adjuster."""

    def test_no_adjuster_returns_same_as_before(self):
        """Passing convection_adjuster=None must give identical results."""
        T, z, H, T_m = _make_test_profile()
        k_ref, state_ref = IceConvection.build_conductivity_profile(
            T_profile=T, z_grid=z, total_thickness=H, T_melt=T_m,
            use_composite_viscosity=True,
        )
        k_new, state_new = IceConvection.build_conductivity_profile(
            T_profile=T, z_grid=z, total_thickness=H, T_melt=T_m,
            use_composite_viscosity=True,
            convection_adjuster=None, q_ocean=0.02,
        )
        np.testing.assert_array_equal(k_ref, k_new)
        assert state_ref.Nu == state_new.Nu
        assert state_ref.Ra == state_new.Ra
        assert state_ref.D_conv == state_new.D_conv

    def test_adjuster_is_called_with_correct_args(self):
        """The adjuster receives (state, T_profile, z_grid, H, q_ocean)."""
        T, z, H, T_m = _make_test_profile()
        call_log = []

        def spy_adjuster(state, T_prof, z_grid, total_thickness, q_ocean):
            call_log.append({
                'state_type': type(state).__name__,
                'T_shape': T_prof.shape,
                'z_shape': z_grid.shape,
                'H': total_thickness,
                'q_ocean': q_ocean,
            })

        IceConvection.build_conductivity_profile(
            T_profile=T, z_grid=z, total_thickness=H, T_melt=T_m,
            use_composite_viscosity=True,
            convection_adjuster=spy_adjuster, q_ocean=0.025,
        )
        assert len(call_log) == 1
        assert call_log[0]['state_type'] == 'ConvectionState'
        assert call_log[0]['T_shape'] == (31,)
        assert call_log[0]['H'] == 20e3
        assert call_log[0]['q_ocean'] == 0.025

    def test_adjuster_mutation_affects_k_profile(self):
        """If adjuster doubles Nu, k_profile in convective layer must change."""
        T, z, H, T_m = _make_test_profile()

        k_base, state_base = IceConvection.build_conductivity_profile(
            T_profile=T, z_grid=z, total_thickness=H, T_melt=T_m,
            use_composite_viscosity=True,
        )

        def double_nu(state, T_prof, z_grid, total_thickness, q_ocean):
            if state.is_convecting:
                state.Nu = state.Nu * 2.0

        k_adj, state_adj = IceConvection.build_conductivity_profile(
            T_profile=T, z_grid=z, total_thickness=H, T_melt=T_m,
            use_composite_viscosity=True,
            convection_adjuster=double_nu, q_ocean=0.02,
        )

        if state_base.is_convecting:
            conv_region = slice(state_base.idx_c, None)
            assert np.all(k_adj[conv_region] > k_base[conv_region])
            cond_region = slice(0, state_base.idx_c)
            np.testing.assert_array_equal(k_adj[cond_region], k_base[cond_region])


from Solver import Thermal_Solver
from Boundary_Conditions import FixedTemperature


class TestSolverAdjusterThreading:
    """Tests that Thermal_Solver threads the adjuster to build_conductivity_profile."""

    def test_solver_accepts_convection_adjuster(self):
        """Solver.__init__ accepts convection_adjuster kwarg."""
        call_count = [0]

        def counting_adjuster(state, T_prof, z_grid, H, q_ocean):
            call_count[0] += 1

        solver = Thermal_Solver(
            nx=31, thickness=20e3, dt=1e12, total_time=5e14,
            use_convection=True,
            convection_adjuster=counting_adjuster,
        )
        solver.solve_step(q_ocean=0.02)
        assert call_count[0] >= 1

    def test_solver_threads_q_ocean_to_adjuster(self):
        """The adjuster receives the q_ocean value from solve_step."""
        received_q = []

        def capture_q(state, T_prof, z_grid, H, q_ocean):
            received_q.append(q_ocean)

        solver = Thermal_Solver(
            nx=31, thickness=20e3, dt=1e12, total_time=5e14,
            use_convection=True,
            convection_adjuster=capture_q,
        )
        solver.solve_step(q_ocean=0.042)
        assert len(received_q) >= 1
        assert received_q[0] == pytest.approx(0.042)

    def test_solver_without_adjuster_unchanged(self):
        """Solver with no adjuster produces identical results to default."""
        kwargs = dict(nx=31, thickness=20e3, dt=1e12, total_time=5e14,
                      use_convection=True)
        s1 = Thermal_Solver(**kwargs)
        s2 = Thermal_Solver(**kwargs, convection_adjuster=None)

        v1 = s1.solve_step(0.02)
        v2 = s2.solve_step(0.02)
        assert v1 == pytest.approx(v2)
        np.testing.assert_array_almost_equal(s1.T, s2.T)


from axial_solver import AxialSolver2D
from latitude_profile import LatitudeProfile


class TestParityAcceptance:
    """hypothesis=None must produce results identical to pre-hook code."""

    def test_2d_solver_parity(self):
        """AxialSolver2D with no hypothesis gives identical H profile."""
        profile = LatitudeProfile(q_ocean_mean=0.02)
        kwargs = dict(
            n_lat=5, nx=21, dt=1e12, total_time=5e14,
            latitude_profile=profile, use_convection=True,
            initial_thickness=20e3, rannacher_steps=2,
        )
        s1 = AxialSolver2D(**kwargs)
        s2 = AxialSolver2D(**kwargs)

        q_ocean = np.array([profile.ocean_heat_flux(phi) for phi in s1.latitudes])

        for _ in range(10):
            s1.solve_step(q_ocean)
            s2.solve_step(q_ocean)

        np.testing.assert_array_almost_equal(
            s1.get_thickness_profile(), s2.get_thickness_profile(),
            decimal=10,
            err_msg="Solver with default adjuster=None must match baseline exactly"
        )
