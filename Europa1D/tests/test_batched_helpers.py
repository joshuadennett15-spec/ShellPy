import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pytest
from constants import Thermal, Planetary, Rheology, HeatFlux
from Physics import IcePhysics


class TestThermophysicalHelpers:
    TEMPS = np.array([100.0, 150.0, 200.0, 250.0, 273.0])

    def test_conductivity(self):
        from batched_solver import _conductivity
        for T in self.TEMPS:
            expected = float(Thermal.conductivity(T))
            result = _conductivity(T)
            assert abs(result - expected) < 1e-12, f"k({T}): {result} != {expected}"

    def test_specific_heat(self):
        from batched_solver import _specific_heat
        for T in self.TEMPS:
            expected = float(Thermal.specific_heat(T))
            result = _specific_heat(T)
            assert abs(result - expected) < 1e-12

    def test_density_ice(self):
        from batched_solver import _density_ice
        for T in self.TEMPS:
            expected = float(Thermal.density_ice(T))
            result = _density_ice(T)
            assert abs(result - expected) < 1e-10

    def test_basal_melting_point(self):
        from batched_solver import _basal_melting_point
        for H in [5e3, 20e3, 50e3, 100e3]:
            expected = float(IcePhysics.basal_melting_point(H))
            result = _basal_melting_point(H, Planetary.GRAVITY)
            assert abs(result - expected) < 1e-10, f"T_melt({H}): {result} != {expected}"

    def test_effective_conductivity_bare(self):
        from batched_solver import _effective_conductivity
        for T in self.TEMPS:
            expected = 612.0 / T
            result = _effective_conductivity(T, 0.0, 0.0, 1.0, 150.0)
            assert abs(result - expected) < 1e-12

    def test_effective_conductivity_porous(self):
        from batched_solver import _effective_conductivity
        T, por, T_phi = 120.0, 0.1, 150.0
        expected = float(IcePhysics.effective_conductivity(T, porosity=por, porosity_cure_temp=T_phi))
        result = _effective_conductivity(T, por, 0.0, 1.0, T_phi)
        assert abs(result - expected) < 1e-12

    def test_effective_conductivity_salt(self):
        from batched_solver import _effective_conductivity
        T, f_s, B_k = 200.0, 0.03, 2.0
        expected = float(IcePhysics.effective_conductivity(T, salt_fraction=f_s, salt_scaling_factor=B_k))
        result = _effective_conductivity(T, 0.0, f_s, B_k, 150.0)
        assert abs(result - expected) < 1e-12


class TestViscosityTidalHelpers:

    def test_viscosity_simple(self):
        from batched_solver import _viscosity_simple
        T = 250.0
        eta_ref = 5.0e13
        Q_v = Rheology.ACTIVATION_ENERGY_V
        R = Rheology.GAS_CONSTANT
        T_melt = 273.0
        expected = float(IcePhysics.viscosity_simple(T, eta_ref, T_melt))
        result = _viscosity_simple(T, eta_ref, Q_v, R, T_melt)
        assert abs(result - expected) / expected < 1e-12

    def test_composite_viscosity(self):
        from batched_solver import _composite_viscosity
        T = 250.0
        expected = float(IcePhysics.composite_viscosity(T))
        result = _composite_viscosity(
            T, Rheology.GRAIN_SIZE, Rheology.GRAIN_WIDTH,
            Rheology.D0V_MEAN, Rheology.D0B_MEAN,
            Rheology.ACTIVATION_ENERGY_V, Rheology.ACTIVATION_ENERGY_B,
            Rheology.MOLAR_VOLUME, Rheology.GAS_CONSTANT,
        )
        assert abs(result - expected) / expected < 1e-10

    def test_composite_viscosity_clipping(self):
        from batched_solver import _composite_viscosity
        R = Rheology.GAS_CONSTANT
        result_cold = _composite_viscosity(
            50.0, 1e-3, 7.13e-10, 9.1e-4, 8.4e-4, 59400.0, 49000.0, 1.97e-5, R)
        result_hot = _composite_viscosity(
            273.0, 1e-3, 7.13e-10, 9.1e-4, 8.4e-4, 59400.0, 49000.0, 1.97e-5, R)
        assert result_cold <= 1e25
        assert result_hot >= 1e12

    def test_tidal_maxwell(self):
        from batched_solver import _tidal_heating_maxwell, _viscosity_simple
        T = 250.0
        eta = _viscosity_simple(T, 5e13, Rheology.ACTIVATION_ENERGY_V,
                                Rheology.GAS_CONSTANT, 273.0)
        result = _tidal_heating_maxwell(HeatFlux.TIDAL_STRAIN, Planetary.ORBITAL_FREQ,
                                        eta, Rheology.RIGIDITY_ICE)
        # For Maxwell comparison, compute expected directly from the Maxwell formula:
        omega = Planetary.ORBITAL_FREQ
        mu = Rheology.RIGIDITY_ICE
        eps0 = HeatFlux.TIDAL_STRAIN
        expected = (eps0**2 * omega**2 * eta) / (2.0 * (1.0 + (omega**2 * eta**2 / mu**2)))
        assert abs(result - expected) / max(expected, 1e-30) < 1e-10

    def test_tidal_andrade(self):
        """Compare Andrade JIT helper against original IcePhysics.tidal_heating.
        Current config has Rheology.MODEL='Andrade' so the comparison is valid."""
        from batched_solver import _tidal_heating_andrade, _viscosity_simple
        from scipy.special import gamma
        assert Rheology.MODEL == "Andrade", "Test requires Andrade config"
        T = 250.0
        eta = _viscosity_simple(T, 5e13, Rheology.ACTIVATION_ENERGY_V,
                                Rheology.GAS_CONSTANT, 273.0)
        gamma_val = gamma(1 + Rheology.ANDRADE_ALPHA)
        result = _tidal_heating_andrade(
            HeatFlux.TIDAL_STRAIN, Planetary.ORBITAL_FREQ, eta,
            Rheology.RIGIDITY_ICE, Rheology.ANDRADE_ALPHA,
            Rheology.ANDRADE_ZETA, gamma_val)
        expected = float(IcePhysics.tidal_heating(
            T, epsilon_0=HeatFlux.TIDAL_STRAIN, use_composite_viscosity=False))
        assert abs(result - expected) / max(expected, 1e-30) < 1e-10


class TestConvectionHelpers:

    def test_howell_Tc(self):
        from batched_solver import _howell_Tc
        from Convection import IceConvection
        from constants import Rheology
        Q_v = 59400.0
        T_melt = 272.0
        expected = IceConvection.howell_cond_base_temp(T_melt, Q_v)
        result = _howell_Tc(T_melt, Q_v, Rheology.GAS_CONSTANT)
        assert abs(result - expected) < 1e-10

    def test_deschamps_Ti(self):
        from batched_solver import _deschamps_Ti
        from Convection import IceConvection
        from constants import Rheology
        T_melt, T_surf, Q_v = 272.0, 104.0, 59400.0
        expected = IceConvection.deschamps_interior_temp(T_melt, T_surf, Q_v)
        result = _deschamps_Ti(T_melt, T_surf, Q_v, Rheology.GAS_CONSTANT, 1.43, -0.03)
        assert abs(result - expected) < 1e-10

    def test_transition_helpers_fallback_for_invalid_inputs(self):
        from batched_solver import _howell_Tc, _deschamps_Ti
        from Convection import IceConvection
        from constants import Rheology

        Tc_py = IceConvection.howell_cond_base_temp(272.0, np.nan)
        Tc_nb = _howell_Tc(272.0, np.nan, Rheology.GAS_CONSTANT)
        assert np.isfinite(Tc_py)
        assert np.isfinite(Tc_nb)

        Ti_py = IceConvection.deschamps_interior_temp(272.0, 104.0, np.nan)
        Ti_nb = _deschamps_Ti(272.0, 104.0, np.nan, Rheology.GAS_CONSTANT, 1.43, -0.03)
        assert np.isfinite(Ti_py)
        assert np.isfinite(Ti_nb)

    def test_green_Tc_Ti(self):
        from batched_solver import _green_Tc_Ti
        from Convection import IceConvection
        from constants import Rheology
        T_melt, T_surf, Q_v = 272.0, 104.0, 59400.0
        eta_ref = 5e13
        expected_Tc, expected_Ti = IceConvection.green_cond_base_temp(
            T_melt, T_surf, Q_v, eta_ref)
        Tc, Ti = _green_Tc_Ti(T_melt, T_surf, Q_v, Rheology.GAS_CONSTANT, eta_ref, 2.24)
        assert abs(Tc - expected_Tc) < 1e-8
        assert abs(Ti - expected_Ti) < 1e-8

    def test_green_Tc_Ti_composite_transition_closure(self):
        from batched_solver import _green_Tc_Ti
        from Convection import IceConvection
        from constants import Rheology
        T_melt, T_surf, Q_v = 272.0, 104.0, 59400.0
        eta_ref = 5e13
        expected_Tc, expected_Ti = IceConvection.green_cond_base_temp(
            T_melt, T_surf, Q_v, eta_ref,
            use_composite_transition_closure=True,
            d_grain=1e-3, d_del=7.13e-10, D0v=9.1e-4, D0b=8.4e-4,
            Q_b=49000.0, p_grain=2.0,
        )
        Tc, Ti = _green_Tc_Ti(
            T_melt, T_surf, Q_v, Rheology.GAS_CONSTANT, eta_ref, 2.24,
            True, 1e-3, 7.13e-10, 9.1e-4, 8.4e-4, 49000.0, 1.97e-5, 2.0,
        )
        assert abs(Tc - expected_Tc) < 1e-8
        assert abs(Ti - expected_Ti) < 1e-8

    def test_rayleigh_number(self):
        from batched_solver import _rayleigh_number, _viscosity_simple
        from Convection import IceConvection
        from constants import Planetary, Rheology
        DT, d, T_mean = 20.0, 10e3, 260.0
        # Direct comparison: compute Ra from both sides
        eta_val = _viscosity_simple(T_mean, 5e13, Rheology.ACTIVATION_ENERGY_V,
                                    Rheology.GAS_CONSTANT, 273.0)
        result = _rayleigh_number(DT, d, T_mean, eta_val, Planetary.GRAVITY, 1.6e-4)
        # Verify Ra is physically reasonable (positive, non-zero for these params)
        assert result > 0
        # Cross-check with IceConvection using simple viscosity
        expected = IceConvection.rayleigh_number(
            DT, d, T_mean, use_composite_viscosity=False, eta_ref=5e13)
        assert abs(result - expected) / expected < 1e-6

    def test_nusselt_simple_subcritical(self):
        from batched_solver import _nusselt_simple
        assert _nusselt_simple(500.0, 1000.0, 0.3446, 0.333) == 1.0

    def test_nusselt_simple_supercritical(self):
        from batched_solver import _nusselt_simple
        from Convection import IceConvection
        Ra = 1e6
        expected = float(IceConvection.nusselt_number(Ra))
        result = _nusselt_simple(Ra, 1000.0, 0.3446, 0.333)
        assert abs(result - expected) / expected < 1e-10

    def test_nusselt_green_subcritical(self):
        from batched_solver import _nusselt_green
        result = _nusselt_green(500.0, 260.0, 240.0, 30.0, 1000.0, 0.3446, 0.333, 1.333)
        assert result == 1.0

    def test_nusselt_green_supercritical(self):
        from batched_solver import _nusselt_green
        from Convection import IceConvection
        Ra, Ti, Tc, DT = 1e6, 260.0, 240.0, 30.0
        expected = IceConvection.nusselt_number_green(Ra, Ti, Tc, DT)
        result = _nusselt_green(Ra, Ti, Tc, DT, 1000.0, 0.3446, 0.333, 1.333)
        assert abs(result - expected) / expected < 1e-10

    def test_harmonic_mean(self):
        from batched_solver import _harmonic_mean
        from Convection import IceConvection
        k = np.array([5.0, 3.0, 1.0, 2.0, 4.0])
        k_half = np.empty(4)
        _harmonic_mean(k, k_half, 5)
        expected = IceConvection.harmonic_mean_vectorized(k)
        np.testing.assert_allclose(k_half, expected, atol=1e-14)

    def test_scan_profile_idx_c_zero(self):
        """Entire shell warm (T[0] >= Tc) should give idx_c=0, D_cond~0."""
        from batched_solver import _scan_profile, _green_Tc_Ti
        nz = 31
        T_melt = 272.0
        T_surf = 271.5
        # Verify Tc < T_surf so every node exceeds Tc
        Tc_check, _ = _green_Tc_Ti(T_melt, T_surf, 59400.0, 8.314, 5e13, 2.24)
        T_col = np.linspace(T_surf, T_melt, nz)
        assert T_col[0] >= Tc_check, f"T[0]={T_col[0]} < Tc={Tc_check}"
        H = 30e3
        idx_c, z_c, D_cond, D_conv, Tc, Ra, Nu, is_conv = _scan_profile(
            T_col, nz, H, T_melt, T_surf, 59400.0, 49000.0, 1e-3,
            7.13e-10, 9.1e-4, 8.4e-4, 1.97e-5,
            5e13, 8.314, 2.24, 1.43, -0.03, 1000.0, 0.3446, 0.333, 1.333,
            1.6e-4, 1.315)
        assert idx_c == 0
        assert D_cond < 1.0
        assert D_conv > 0.0

    def test_scan_profile_no_convection(self):
        """Cold linear profile below Tc should return non-convecting."""
        from batched_solver import _scan_profile
        nz = 31
        T_col = np.linspace(104.0, 200.0, nz)
        H = 10e3
        T_melt = 272.0
        idx_c, z_c, D_cond, D_conv, Tc, Ra, Nu, is_conv = _scan_profile(
            T_col, nz, H, T_melt, 104.0, 59400.0, 49000.0, 1e-3,
            7.13e-10, 9.1e-4, 8.4e-4, 1.97e-5,
            5e13, 8.314, 2.24, 1.43, -0.03, 1000.0, 0.3446, 0.333, 1.333,
            1.6e-4, 1.315)
        assert not is_conv
        assert D_conv == 0.0

    def test_scan_profile_vs_original(self):
        """Warm profile should match IceConvection.scan_temperature_profile."""
        from batched_solver import _scan_profile, NU_SCALING_GREEN
        from Convection import IceConvection
        from Physics import IcePhysics
        nz = 31
        H = 30e3
        T_melt = float(IcePhysics.basal_melting_point(H))
        T_col = np.linspace(104.0, T_melt, nz)
        z_grid = np.linspace(0, H, nz)
        state = IceConvection.scan_temperature_profile(
            T_col, z_grid, H, T_melt, Q_v=59400.0, Q_b=49000.0,
            d_grain=1e-3, use_composite_viscosity=True, eta_ref=5e13,
            nu_scaling="green")
        # nu_scaling_id=NU_SCALING_GREEN so batched path uses FK viscosity,
        # matching the Python path's calibration-consistent Ra computation.
        idx_c, z_c, D_cond, D_conv, Tc, Ra, Nu, is_conv = _scan_profile(
            T_col, nz, H, T_melt, 104.0, 59400.0, 49000.0, 1e-3,
            7.13e-10, 9.1e-4, 8.4e-4, 1.97e-5,
            5e13, 8.314, 2.24, 1.43, -0.03, 1000.0, 0.3446, 0.333, 1.333,
            1.6e-4, 1.315, False, 2.0, NU_SCALING_GREEN)
        assert abs(z_c - state.z_c) < 1.0
        assert abs(D_cond - state.D_cond) < 1.0
        if state.Ra > 0:
            assert abs(Ra - state.Ra) / state.Ra < 0.01

    def test_scan_profile_vs_original_composite_transition_closure(self):
        """Composite transition closure should match the Python scan path."""
        from batched_solver import _scan_profile, NU_SCALING_GREEN
        from Convection import IceConvection
        from Physics import IcePhysics
        nz = 31
        H = 30e3
        T_melt = float(IcePhysics.basal_melting_point(H))
        T_col = np.linspace(104.0, T_melt, nz)
        z_grid = np.linspace(0, H, nz)
        state = IceConvection.scan_temperature_profile(
            T_col, z_grid, H, T_melt, Q_v=59400.0, Q_b=49000.0,
            d_grain=1e-3, p_grain=2.0, use_composite_viscosity=True, eta_ref=5e13,
            use_composite_transition_closure=True, d_del=7.13e-10, D0v=9.1e-4, D0b=8.4e-4,
            nu_scaling="green",
        )
        idx_c, z_c, D_cond, D_conv, Tc, Ra, Nu, is_conv = _scan_profile(
            T_col, nz, H, T_melt, 104.0, 59400.0, 49000.0, 1e-3,
            7.13e-10, 9.1e-4, 8.4e-4, 1.97e-5,
            5e13, 8.314, 2.24, 1.43, -0.03, 1000.0, 0.3446, 0.333, 1.333,
            1.6e-4, 1.315, True, 2.0, NU_SCALING_GREEN)
        assert abs(Tc - state.T_c) < 1e-8
        assert abs(z_c - state.z_c) < 1.0
        assert abs(D_cond - state.D_cond) < 1.0
        assert abs(D_conv - state.D_conv) < 1.0
        if state.Ra > 0:
            assert abs(Ra - state.Ra) / state.Ra < 0.01


class TestNuScalingSelector:

    def test_invalid_nu_scaling_raises(self):
        """Invalid NU_SCALING value must raise ValueError at import time."""
        import json, subprocess
        config_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'config.json')
        with open(config_path) as f:
            original = json.load(f)
        try:
            modified = json.loads(json.dumps(original))
            modified['convection']['NU_SCALING'] = 'bogus'
            with open(config_path, 'w') as f:
                json.dump(modified, f)
            src_dir = os.path.join(os.path.dirname(__file__), '..', 'src')
            result = subprocess.run(
                [sys.executable, '-c', 'from constants import Convection'],
                capture_output=True, text=True, cwd=src_dir,
            )
            assert result.returncode != 0, "Should have failed on invalid NU_SCALING"
            assert 'ValueError' in result.stderr
            assert 'NU_SCALING' in result.stderr
        finally:
            with open(config_path, 'w') as f:
                json.dump(original, f, indent=4)

    def test_dv2021_is_accepted(self):
        """dv2021 scaling is a valid NU_SCALING value (now implemented)."""
        import json, subprocess
        config_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'config.json')
        with open(config_path) as f:
            original = json.load(f)
        try:
            modified = json.loads(json.dumps(original))
            modified['convection']['NU_SCALING'] = 'dv2021'
            with open(config_path, 'w') as f:
                json.dump(modified, f)
            src_dir = os.path.join(os.path.dirname(__file__), '..', 'src')
            result = subprocess.run(
                [sys.executable, '-c', 'from constants import Convection'],
                capture_output=True, text=True, cwd=src_dir,
            )
            assert result.returncode == 0, f"dv2021 should be accepted: {result.stderr}"
        finally:
            with open(config_path, 'w') as f:
                json.dump(original, f, indent=4)


class TestThomasAndStefan:

    def test_thomas_solve_identity(self):
        """Thomas solver on trivial system: I*x = b -> x = b."""
        from batched_solver import _thomas_solve
        n = 5
        a = np.zeros(n)
        b = np.ones(n)
        c = np.zeros(n)
        d = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        x = np.empty(n)
        _thomas_solve(a.copy(), b.copy(), c.copy(), d.copy(), x, n)
        np.testing.assert_allclose(x, [1.0, 2.0, 3.0, 4.0, 5.0], atol=1e-14)

    def test_thomas_solve_vs_scipy(self):
        """Compare Thomas vs scipy.linalg.solve_banded."""
        from batched_solver import _thomas_solve
        from scipy.linalg import solve_banded
        n = 10
        np.random.seed(42)
        a_lower = np.random.randn(n)
        a_main = np.random.randn(n) + 5  # diagonally dominant
        a_upper = np.random.randn(n)
        rhs = np.random.randn(n)
        a_lower[n - 1] = 0.0
        a_upper[n - 1] = 0.0

        # scipy banded solve
        ab = np.zeros((3, n))
        ab[0, 1:] = a_upper[:n - 1]
        ab[1, :] = a_main
        ab[2, :-1] = a_lower[:n - 1]
        x_scipy = solve_banded((1, 1), ab, rhs.copy())

        # Thomas solve
        x_thomas = np.empty(n)
        _thomas_solve(a_lower.copy(), a_main.copy(), a_upper.copy(), rhs.copy(), x_thomas, n)
        np.testing.assert_allclose(x_thomas, x_scipy, atol=1e-10)

    def test_thomas_boundary_rows(self):
        """Test with boundary rows (diag=1, off-diag=0) like the actual solver."""
        from batched_solver import _thomas_solve
        n = 5
        a = np.array([0.0, -0.3, -0.3, 0.0, 0.0])  # boundary: a[3]=0 (basal)
        b = np.array([1.0, 1.6, 1.6, 1.6, 1.0])     # boundaries: diag=1
        c = np.array([0.0, -0.3, -0.3, -0.3, 0.0])   # boundary: c[0]=0 (surface)
        d = np.array([104.0, 150.0, 200.0, 250.0, 273.0])
        x = np.empty(n)
        _thomas_solve(a.copy(), b.copy(), c.copy(), d.copy(), x, n)
        # Verify boundary rows: x[0] = 104, x[4] = 273
        assert abs(x[0] - 104.0) < 1e-10
        assert abs(x[4] - 273.0) < 1e-10

    def test_stefan_velocity(self):
        """Stefan velocity vs original IcePhysics.stefan_velocity."""
        from batched_solver import _stefan_velocity
        nz = 31
        H = 20e3
        dz = H / (nz - 1)
        T_melt = float(IcePhysics.basal_melting_point(H))
        T_col = np.linspace(104.0, T_melt, nz)
        q_ocean = 0.02
        k_basal = float(IcePhysics.effective_conductivity(T_col[-1]))
        expected = float(IcePhysics.stefan_velocity(T_col, dz, q_ocean, k_basal))
        result = _stefan_velocity(T_col, nz, dz, q_ocean)
        assert abs(result - expected) / abs(expected) < 1e-10
