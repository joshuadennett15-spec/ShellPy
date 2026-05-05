"""
Tests for Deschamps & Vilella (2021) mixed-heating stagnant-lid scaling.

Validates the 4 DV2021 helper methods in Convection.py against
analytical expectations from the paper (JGR Planets, Table 2).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pytest
from Convection import IceConvection


class TestDV2021InteriorTemperature:
    """Eq. 21: T_m_tilde root-finding."""

    def test_pure_bottom_heating_analytical(self):
        """H_tilde=0 => T_m = 1 - a1/gamma (no Ra dependence)."""
        gamma = np.log(1e4)  # ~9.21
        a1 = 1.23
        expected = 1.0 - a1 / gamma  # 0.8665

        T_m = IceConvection.dv2021_interior_temperature(
            Ra_surf=16.0, gamma=gamma, H_tilde=0.0,
        )
        assert abs(T_m - expected) < 1e-6, f"T_m={T_m:.6f}, expected={expected:.6f}"

    def test_pure_bottom_heating_ra_independent(self):
        """With H=0, T_m should not depend on Ra_surf."""
        gamma = np.log(1e4)
        T_m_lo = IceConvection.dv2021_interior_temperature(
            Ra_surf=10.0, gamma=gamma, H_tilde=0.0,
        )
        T_m_hi = IceConvection.dv2021_interior_temperature(
            Ra_surf=1e8, gamma=gamma, H_tilde=0.0,
        )
        assert abs(T_m_lo - T_m_hi) < 1e-10

    def test_internal_heating_raises_temperature(self):
        """Positive H_tilde should increase T_m above the H=0 baseline."""
        gamma = np.log(1e4)
        T_m_base = IceConvection.dv2021_interior_temperature(
            Ra_surf=1e4, gamma=gamma, H_tilde=0.0,
        )
        T_m_heated = IceConvection.dv2021_interior_temperature(
            Ra_surf=1e4, gamma=gamma, H_tilde=5.0,
        )
        assert T_m_heated > T_m_base

    def test_gt1_regime_coefficients(self):
        """Ur>1 regime uses different coefficients (c3=1/3, c4=1.72)."""
        gamma = np.log(1e4)
        T_m_lt1 = IceConvection.dv2021_interior_temperature(
            Ra_surf=1e4, gamma=gamma, H_tilde=5.0, ur_regime="lt1",
        )
        T_m_gt1 = IceConvection.dv2021_interior_temperature(
            Ra_surf=1e4, gamma=gamma, H_tilde=5.0, ur_regime="gt1",
        )
        # They should differ since coefficients differ
        assert T_m_lt1 != pytest.approx(T_m_gt1, abs=1e-4)


class TestDV2021SurfaceHeatFlux:
    """Eq. 23: Phi_top = a * Ra_eff^b / gamma^c."""

    def test_increases_with_ra_eff(self):
        """Phi_top must be monotonically increasing with Ra_eff."""
        gamma = np.log(1e5)
        Ra_values = [1e4, 1e6, 1e8, 1e10]
        fluxes = [
            IceConvection.dv2021_surface_heat_flux(Ra, gamma, ur_regime="lt1")
            for Ra in Ra_values
        ]
        for i in range(len(fluxes) - 1):
            assert fluxes[i + 1] > fluxes[i], (
                f"Phi_top should increase: {fluxes[i+1]:.4f} <= {fluxes[i]:.4f}"
            )

    def test_known_value_lt1(self):
        """Spot-check: Ra_eff=1e6, gamma=10, Ur<1 coefficients, no FK correction."""
        # Phi_top = 1.46 * (1e6)^0.27 / 10^1.21
        # = 1.46 * 28.84 / 16.22 = 2.596
        Ra_eff = 1e6
        gamma = 10.0
        expected_raw = 1.46 * Ra_eff**0.27 / gamma**1.21

        from ConfigManager import ConfigManager
        fk = ConfigManager.get("convection", "FK_CORRECTION", False)
        fk_factor = ConfigManager.get("convection", "FK_CORRECTION_FACTOR", 0.75)
        expected = expected_raw * fk_factor if fk else expected_raw

        result = IceConvection.dv2021_surface_heat_flux(Ra_eff, gamma, ur_regime="lt1")
        assert result == pytest.approx(expected, rel=1e-6)


class TestDV2021LidThickness:
    """Eq. 26: d_lid = a_lid * gamma^c / Ra_eff^b."""

    def test_decreases_with_ra_eff(self):
        """d_lid must be monotonically decreasing with Ra_eff."""
        gamma = np.log(1e5)
        Ra_values = [1e4, 1e6, 1e8, 1e10]
        lids = [
            IceConvection.dv2021_lid_thickness(Ra, gamma, ur_regime="lt1")
            for Ra in Ra_values
        ]
        for i in range(len(lids) - 1):
            assert lids[i + 1] < lids[i], (
                f"d_lid should decrease: {lids[i+1]:.6f} >= {lids[i]:.6f}"
            )

    def test_known_value_lt1(self):
        """Spot-check: Ra_eff=1e6, gamma=10, Ur<1 coefficients."""
        Ra_eff = 1e6
        gamma = 10.0
        expected = 0.633 * gamma**1.21 / Ra_eff**0.27

        result = IceConvection.dv2021_lid_thickness(Ra_eff, gamma, ur_regime="lt1")
        assert result == pytest.approx(expected, rel=1e-6)

    def test_gt1_thicker_lid(self):
        """Ur>1 has a_lid=0.667 > 0.633, so lid is thicker for same Ra_eff."""
        gamma = 10.0
        Ra_eff = 1e6
        d_lt1 = IceConvection.dv2021_lid_thickness(Ra_eff, gamma, ur_regime="lt1")
        d_gt1 = IceConvection.dv2021_lid_thickness(Ra_eff, gamma, ur_regime="gt1")
        assert d_gt1 > d_lt1


class TestDV2021Solve:
    """Full solve with regime switching (Section 5)."""

    def test_low_H_stays_lt1(self):
        """Pure bottom heating stays in Ur<1 regime."""
        result = IceConvection.dv2021_solve(
            Ra_surf=1e6, gamma=np.log(1e5), H_tilde=0.0,
        )
        assert result["regime"] == "lt1"
        assert result["Phi_bot"] > 0

    def test_high_H_triggers_gt1(self):
        """Strong internal heating switches to Ur>1 regime."""
        result = IceConvection.dv2021_solve(
            Ra_surf=1e6, gamma=np.log(1e5), H_tilde=100.0,
        )
        assert result["regime"] == "gt1"

    def test_high_H_phi_bot_clamped(self):
        """When Ur>1 gives Phi_bot>0 (contradiction), it is clamped to 0."""
        result = IceConvection.dv2021_solve(
            Ra_surf=1e6, gamma=np.log(1e5), H_tilde=100.0,
        )
        assert result["Phi_bot"] == pytest.approx(0.0)

    def test_output_keys(self):
        """Solve returns all expected keys."""
        result = IceConvection.dv2021_solve(
            Ra_surf=1e4, gamma=10.0, H_tilde=0.0,
        )
        expected_keys = {"T_m_tilde", "Ra_eff", "Phi_top", "Phi_bot", "d_lid_nd", "regime"}
        assert set(result.keys()) == expected_keys

    def test_ra_eff_consistency(self):
        """Ra_eff = Ra_surf * exp(gamma * T_m) must hold."""
        Ra_surf = 1e4
        gamma = 10.0
        result = IceConvection.dv2021_solve(Ra_surf, gamma, H_tilde=0.0)
        expected_ra = Ra_surf * np.exp(gamma * result["T_m_tilde"])
        assert result["Ra_eff"] == pytest.approx(expected_ra, rel=1e-8)

    def test_europa_like_d_lid_range(self):
        """Europa-like parameters give d_lid in 5-40 km range.

        Uses Ra_surf=500, gamma=8 with D_shell=30 km to represent a
        weakly convecting Europa ice shell near onset conditions.
        """
        D_shell = 30e3  # 30 km total shell
        gamma = 8.0
        Ra_surf = 500.0

        result = IceConvection.dv2021_solve(Ra_surf, gamma, H_tilde=0.0)
        d_lid_km = result["d_lid_nd"] * D_shell / 1e3

        assert 5.0 <= d_lid_km <= 40.0, (
            f"d_lid={d_lid_km:.1f} km outside Europa range [5, 40] km"
        )
