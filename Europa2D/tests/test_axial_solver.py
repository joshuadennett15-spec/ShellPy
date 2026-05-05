import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import src  # triggers import path setup

import numpy as np
import pytest
from latitude_profile import LatitudeProfile
from axial_solver import AxialSolver2D


def _make_params():
    return {
        'd_grain': 1e-3, 'Q_v': 59.4e3, 'Q_b': 49.0e3,
        'mu_ice': 3.3e9, 'D0v': 9.1e-4, 'D0b': 8.4e-4,
        'd_del': 7e-10, 'f_porosity': 0.0, 'f_salt': 0.0,
        'B_k': 1.0, 'T_phi': 150.0,
    }


def _make_profile():
    return LatitudeProfile(T_eq=110.0, epsilon_eq=6e-6, epsilon_pole=1.2e-5, q_ocean_mean=0.02)


class TestAxialSolverInit:
    """Tests for AxialSolver2D construction."""

    def test_creates_correct_number_of_columns(self):
        solver = AxialSolver2D(
            n_lat=19, nx=31, latitude_profile=_make_profile(),
            physics_params=_make_params(),
        )
        assert len(solver.columns) == 19

    def test_latitudes_span_equator_to_pole(self):
        solver = AxialSolver2D(
            n_lat=19, nx=31, latitude_profile=_make_profile(),
            physics_params=_make_params(),
        )
        assert solver.latitudes[0] == pytest.approx(0.0)
        assert solver.latitudes[-1] == pytest.approx(np.pi / 2)

    def test_each_column_has_correct_surface_temp(self):
        profile = _make_profile()
        solver = AxialSolver2D(
            n_lat=5, nx=31, latitude_profile=profile,
            physics_params=_make_params(),
        )
        for j, col in enumerate(solver.columns):
            expected_T = profile.surface_temperature(solver.latitudes[j])
            assert col.T[0] == pytest.approx(expected_T, abs=1.0)

    def test_thickness_profile_returns_array(self):
        solver = AxialSolver2D(
            n_lat=5, nx=31, latitude_profile=_make_profile(),
            physics_params=_make_params(), initial_thickness=20e3,
        )
        H = solver.get_thickness_profile()
        assert H.shape == (5,)
        assert np.all(H == pytest.approx(20e3))

    def test_cold_polar_columns_keep_full_convection_parameterization(self):
        solver = AxialSolver2D(
            n_lat=37, nx=21, latitude_profile=_make_profile(),
            physics_params=_make_params(), use_convection=True,
        )
        pole_col = solver.columns[-1]
        assert pole_col.use_convection is True
        assert pole_col.convection_ramp == pytest.approx(1.0)
        assert pole_col.phys.get('use_composite_transition_closure') is True
        assert pole_col.phys.get('use_onset_consistent_partition') is True


class TestLateralDiffusion:
    """Tests for the explicit lateral heat diffusion step."""

    def _make_solver(self, n_lat=5, nx=11):
        return AxialSolver2D(
            n_lat=n_lat, nx=nx, latitude_profile=_make_profile(),
            physics_params=_make_params(), use_convection=False,
            initial_thickness=20e3,
        )

    def test_uniform_temperature_no_change(self):
        """If all columns have the same T profile, lateral step should do nothing."""
        solver = self._make_solver()
        for col in solver.columns:
            col.T[:] = np.linspace(100, 270, col.nx)
        T_before = np.array([col.T.copy() for col in solver.columns])
        solver._lateral_diffusion_step()
        T_after = np.array([col.T.copy() for col in solver.columns])
        assert np.allclose(T_before, T_after, atol=1e-10)

    def test_hot_column_cools(self):
        """A column hotter than its neighbors should cool from lateral diffusion."""
        solver = self._make_solver(n_lat=5)
        base_T = np.linspace(100, 270, solver.nx)
        for col in solver.columns:
            col.T[:] = base_T.copy()
        solver.columns[2].T[5] += 50.0
        T_mid_before = solver.columns[2].T[5]
        solver._lateral_diffusion_step()
        T_mid_after = solver.columns[2].T[5]
        assert T_mid_after < T_mid_before

    def test_symmetric_boundary_conditions(self):
        """Verify no crash and finite values at boundaries."""
        solver = self._make_solver(n_lat=5)
        base_T = np.linspace(100, 270, solver.nx)
        for col in solver.columns:
            col.T[:] = base_T.copy()
        solver.columns[2].T[5] += 50.0
        solver._lateral_diffusion_step()
        assert np.all(np.isfinite([col.T for col in solver.columns]))


class TestSolveStep:
    """Tests for the full radial+lateral solve step."""

    def _make_solver(self, n_lat=5, nx=21):
        profile = LatitudeProfile(T_eq=110.0, q_ocean_mean=0.02)
        return AxialSolver2D(
            n_lat=n_lat, nx=nx, latitude_profile=profile,
            physics_params=_make_params(), use_convection=False,
            initial_thickness=20e3,
        )

    def test_returns_velocity_array(self):
        solver = self._make_solver()
        q_profile = solver.profile.ocean_heat_flux(solver.latitudes)
        velocities = solver.solve_step(q_profile)
        assert velocities.shape == (5,)
        assert np.all(np.isfinite(velocities))

    def test_thickness_changes_after_step(self):
        solver = self._make_solver()
        H_before = solver.get_thickness_profile().copy()
        q_profile = solver.profile.ocean_heat_flux(solver.latitudes)
        solver.solve_step(q_profile)
        H_after = solver.get_thickness_profile()
        assert not np.allclose(H_before, H_after)


class TestRunToEquilibrium:
    """Tests for run_to_equilibrium convergence."""

    def test_converges_with_conduction_only(self):
        """Pure conduction (no convection) should converge."""
        profile = LatitudeProfile(
            T_eq=110.0, q_ocean_mean=0.02,
            epsilon_eq=0.0, epsilon_pole=0.0,
            ocean_pattern="uniform",
        )
        solver = AxialSolver2D(
            n_lat=3, nx=21, latitude_profile=profile,
            physics_params=_make_params(), use_convection=False,
            initial_thickness=20e3, dt=1e10, rannacher_steps=0,
        )
        result = solver.run_to_equilibrium(
            threshold=1e-10, max_steps=2000, verbose=False
        )
        assert 'H_profile_km' in result
        assert result['converged']
        H = result['H_profile_km']
        assert H.shape == (3,)
        assert np.all(H > 0.5)
        assert np.all(H < 200)


class TestImplicitLateralDiffusion:
    """Tests for the implicit lateral diffusion step."""

    def _make_solver(self, n_lat=19, nx=11, lateral_method='implicit'):
        return AxialSolver2D(
            n_lat=n_lat, nx=nx, latitude_profile=_make_profile(),
            physics_params=_make_params(), use_convection=False,
            initial_thickness=20e3, lateral_method=lateral_method,
        )

    def test_uniform_temperature_no_change(self):
        """If all columns have the same T profile, implicit step does nothing."""
        solver = self._make_solver()
        for col in solver.columns:
            col.T[:] = np.linspace(100, 270, col.nx)
        T_before = np.array([col.T.copy() for col in solver.columns])
        solver._lateral_diffusion_step_implicit()
        T_after = np.array([col.T.copy() for col in solver.columns])
        assert np.allclose(T_before, T_after, atol=1e-10)

    def test_hot_column_cools(self):
        """A column hotter than neighbors should cool (same as explicit)."""
        solver = self._make_solver(n_lat=5)
        base_T = np.linspace(100, 270, solver.nx)
        for col in solver.columns:
            col.T[:] = base_T.copy()
        solver.columns[2].T[5] += 50.0
        T_mid_before = solver.columns[2].T[5]
        solver._lateral_diffusion_step_implicit()
        T_mid_after = solver.columns[2].T[5]
        assert T_mid_after < T_mid_before

    def test_no_nan_or_inf(self):
        """Implicit step should never produce NaN or Inf."""
        solver = self._make_solver()
        base_T = np.linspace(100, 270, solver.nx)
        for col in solver.columns:
            col.T[:] = base_T.copy()
        solver.columns[-1].T[5] += 30.0
        solver._lateral_diffusion_step_implicit()
        T_all = np.array([col.T for col in solver.columns])
        assert np.all(np.isfinite(T_all))

    def test_matches_explicit_for_small_perturbation(self):
        """
        For small alpha (weak diffusion), implicit and explicit should
        agree to O(alpha^2).
        """
        solver_imp = self._make_solver(n_lat=9, lateral_method='implicit')
        solver_exp = self._make_solver(n_lat=9, lateral_method='explicit')

        base_T = np.linspace(100, 270, solver_imp.nx)
        perturbation = 5.0  # small enough that alpha*dT is small
        for s in (solver_imp, solver_exp):
            for col in s.columns:
                col.T[:] = base_T.copy()
            s.columns[4].T[5] += perturbation

        solver_imp._lateral_diffusion_step_implicit()
        solver_exp._lateral_diffusion_step()

        T_imp = np.array([col.T for col in solver_imp.columns])
        T_exp = np.array([col.T for col in solver_exp.columns])

        # They should agree to O(alpha^2); for our parameters alpha ~ 1e-4,
        # so the difference should be negligible relative to the perturbation.
        assert np.allclose(T_imp, T_exp, atol=1e-6)

    def test_pole_lhopital_implicit(self):
        """Verify L'Hopital pole treatment is encoded in the implicit matrix."""
        from constants import Planetary, Thermal
        solver = self._make_solver(n_lat=19)
        R = Planetary.RADIUS

        # Same manufactured test as explicit: T(phi, z_5) = A + B*cos^2(phi)
        base_T = np.linspace(100, 270, solver.nx)
        for col in solver.columns:
            col.T[:] = base_T.copy()

        A, B, test_node = 200.0, 50.0, 5
        for j, col in enumerate(solver.columns):
            phi_j = solver.latitudes[j]
            col.T[test_node] = A + B * np.cos(phi_j) ** 2

        T_pole_before = solver.columns[-1].T[test_node]

        solver._lateral_diffusion_step_implicit()

        T_pole_after = solver.columns[-1].T[test_node]

        # Pole value should move toward its neighbor (which is warmer
        # because cos^2 at the second-to-last latitude > 0)
        T_neighbor_before = A + B * np.cos(solver.latitudes[-2]) ** 2
        assert T_pole_after > T_pole_before  # pole warms toward neighbor
        assert T_pole_after < T_neighbor_before  # but doesn't overshoot

    def test_area_weighted_energy_nearly_conserved(self):
        """
        Implicit diffusion should approximately conserve area-weighted
        thermal energy.  Not exact because alpha_j uses column-mean T,
        so the perturbed column has slightly different k and rho*cp.
        (Same limitation as the explicit method.)
        """
        solver = self._make_solver(n_lat=19)
        base_T = np.linspace(100, 270, solver.nx)
        for col in solver.columns:
            col.T[:] = base_T.copy()
        # Add a localized perturbation
        solver.columns[5].T[3] += 100.0

        cos_phi = np.cos(solver.latitudes)
        T_before = np.array([col.T.copy() for col in solver.columns])
        energy_before = np.sum(T_before * cos_phi[:, None])

        solver._lateral_diffusion_step_implicit()

        T_after = np.array([col.T for col in solver.columns])
        energy_after = np.sum(T_after * cos_phi[:, None])

        # Conserved to ~1e-7 relative; not machine-precision because
        # per-column alpha varies with column-mean temperature.
        assert energy_before == pytest.approx(energy_after, rel=1e-6)

    def test_solve_step_uses_implicit_by_default(self):
        """AxialSolver2D.solve_step should use implicit lateral method by default."""
        solver = self._make_solver(n_lat=5, nx=11)
        assert solver.lateral_method == 'implicit'


class TestPoleBoundaryCondition:
    """Tests for the L'Hopital pole BC using a manufactured profile."""

    def _make_solver(self, n_lat=19, nx=11):
        return AxialSolver2D(
            n_lat=n_lat, nx=nx, latitude_profile=_make_profile(),
            physics_params=_make_params(), use_convection=False,
            initial_thickness=20e3,
        )

    def test_pole_update_matches_lhopital_limit(self):
        """
        Manufactured test: set T(phi) = cos^2(phi) scaled profile.

        The geographic diffusion operator:
            L[T] = (1/(R^2 cos(phi))) d/dphi[cos(phi) dT/dphi]

        For T(phi) = cos^2(phi):
            L[T] = (1/R^2) * (-2 + 6*sin^2(phi))

        At the pole (phi=pi/2): L[T] = 4/R^2

        The L'Hopital-based discretisation should give:
            dT[pole] = 4 * alpha * (T[j-1] - T[j])
        """
        from constants import Planetary, Thermal
        solver = self._make_solver(n_lat=19)
        R = Planetary.RADIUS

        # Set each column to the same T profile (so only pole BC matters)
        base_T = np.linspace(100, 270, solver.nx)
        for col in solver.columns:
            col.T[:] = base_T.copy()

        # Now impose a lateral T variation at one depth node:
        # T(phi, z_5) = A + B*cos^2(phi)
        A = 200.0
        B = 50.0
        test_node = 5
        for j, col in enumerate(solver.columns):
            phi_j = solver.latitudes[j]
            col.T[test_node] = A + B * np.cos(phi_j) ** 2

        # Record T at pole and its neighbor before the step
        j_pole = solver.n_lat - 1
        T_pole_before = solver.columns[j_pole].T[test_node]
        T_neighbor = solver.columns[j_pole - 1].T[test_node]

        # Run lateral diffusion
        solver._lateral_diffusion_step()

        T_pole_after = solver.columns[j_pole].T[test_node]
        dT_actual = T_pole_after - T_pole_before

        # Expected from L'Hopital: dT = 4*alpha*(T[j-1] - T[j])
        dphi = solver.dphi
        T_mean_pole = np.mean(solver.columns[j_pole].T)
        k_pole = float(Thermal.conductivity(T_mean_pole))
        rho_cp_pole = float(Thermal.density_ice(T_mean_pole)) * Thermal.SPECIFIC_HEAT
        _, dt_eff = solver.columns[0]._get_theta_and_dt()
        alpha = k_pole * dt_eff / (rho_cp_pole * R**2 * dphi**2)

        dT_expected = 4.0 * alpha * (T_neighbor - T_pole_before)

        # Should match to within floating-point tolerance.
        # Small O(1e-7) relative error arises because the test recomputes
        # T_mean_pole from scratch while the solver computed it before the
        # cos^2(phi) perturbation was imposed on a single node.
        assert dT_actual == pytest.approx(dT_expected, rel=1e-6), \
            f"Pole BC mismatch: actual={dT_actual:.6e}, expected={dT_expected:.6e}"

    def test_pole_no_nan_or_inf(self):
        """Pole BC should never produce NaN or Inf values."""
        solver = self._make_solver(n_lat=19)
        base_T = np.linspace(100, 270, solver.nx)
        for col in solver.columns:
            col.T[:] = base_T.copy()
        # Add a perturbation
        solver.columns[-1].T[5] += 30.0
        solver._lateral_diffusion_step()
        assert np.all(np.isfinite(solver.columns[-1].T))

