"""
Tests for GBS composite creep model (Goldsby & Kohlstedt 2001).

Covers:
    - gbs_strain_rate: low-T and high-T regimes
    - dislocation_strain_rate: low-T and high-T regimes
    - composite_viscosity with creep_model="composite_gbs"
    - composite_viscosity backward compatibility (diffusion mode)
    - convective_stress order-of-magnitude check
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from Physics import IcePhysics


# =========================================================================
# GBS strain rate (Goldsby & Kohlstedt 2001, Table 5, Eq. 15)
# =========================================================================

class TestGBSStrainRate:
    """Tests for IcePhysics.gbs_strain_rate."""

    def test_gbs_low_temp(self):
        """GBS at T=240K (< 255K threshold), d=1mm, sigma=10kPa.

        Hand-calculated:
          A=6.18e-14, Q=49000, n=1.8, p=1.4
          eps = 6.18e-14 * (1e-3)^(-1.4) * (10000)^1.8 * exp(-49000/(8.314*240))
              = 3.3576e-13 s^-1
        """
        T = 240.0
        d = 1.0e-3   # 1 mm
        sigma = 1.0e4  # 10 kPa
        eps = IcePhysics.gbs_strain_rate(T, d, sigma)
        np.testing.assert_allclose(eps, 3.3576e-13, rtol=1e-3)

    def test_gbs_high_temp(self):
        """GBS at T=260K (>= 255K threshold), d=1mm, sigma=10kPa.

        Hand-calculated:
          A=4.76e15, Q=192000, n=1.8, p=1.4
          eps = 4.76e15 * (1e-3)^(-1.4) * (10000)^1.8 * exp(-192000/(8.314*260))
              = 3.1838e-12 s^-1
        """
        T = 260.0
        d = 1.0e-3
        sigma = 1.0e4
        eps = IcePhysics.gbs_strain_rate(T, d, sigma)
        np.testing.assert_allclose(eps, 3.1838e-12, rtol=1e-3)

    def test_gbs_vectorized(self):
        """GBS strain rate works on numpy arrays spanning both regimes."""
        T = np.array([240.0, 260.0])
        d = 1.0e-3
        sigma = 1.0e4
        eps = IcePhysics.gbs_strain_rate(T, d, sigma)
        assert eps.shape == (2,)
        # Low-T value should be smaller than high-T value
        assert eps[0] < eps[1]


# =========================================================================
# Dislocation strain rate (GK2001, Table 5, Eq. 13)
# =========================================================================

class TestDislocationStrainRate:
    """Tests for IcePhysics.dislocation_strain_rate."""

    def test_dislocation_low_temp(self):
        """Dislocation creep at T=240K (< 258K threshold), sigma=50kPa.

        Hand-calculated:
          A=4.0e-19, Q=60000, n=4.0
          eps = 4.0e-19 * (50000)^4 * exp(-60000/(8.314*240))
              = 2.1818e-13 s^-1
        """
        T = 240.0
        sigma = 5.0e4  # 50 kPa
        eps = IcePhysics.dislocation_strain_rate(T, sigma)
        np.testing.assert_allclose(eps, 2.1818e-13, rtol=1e-3)

    def test_dislocation_high_temp(self):
        """Dislocation creep at T=265K (>= 258K threshold), sigma=50kPa.

        Hand-calculated:
          A=6.0e4, Q=181000, n=4.0
          eps = 6.0e4 * (50000)^4 * exp(-181000/(8.314*265))
              = 7.8614e-13 s^-1
        """
        T = 265.0
        sigma = 5.0e4
        eps = IcePhysics.dislocation_strain_rate(T, sigma)
        np.testing.assert_allclose(eps, 7.8614e-13, rtol=1e-3)

    def test_dislocation_grain_size_independent(self):
        """Dislocation creep has no grain-size dependence (p=0)."""
        T = 240.0
        sigma = 5.0e4
        # The function signature has no grain-size parameter at all
        eps = IcePhysics.dislocation_strain_rate(T, sigma)
        assert np.isfinite(eps)
        assert eps > 0


# =========================================================================
# Composite viscosity: composite_gbs mode
# =========================================================================

class TestCompositeGBS:
    """Tests for composite_viscosity with creep_model='composite_gbs'."""

    def test_composite_gbs_lower_viscosity_than_diffusion(self):
        """At d=1mm, composite_gbs must give LOWER eta than diffusion-only.

        GBS dominates at mm grain sizes, adding extra strain rate channels
        that reduce effective viscosity.
        """
        T = 250.0
        d = 1.0e-3   # 1 mm -- squarely in GBS-dominated regime
        sigma = 1.0e4  # 10 kPa

        eta_diff = IcePhysics.composite_viscosity(T, d_grain=d, creep_model="diffusion")
        eta_comp = IcePhysics.composite_viscosity(
            T, d_grain=d, creep_model="composite_gbs", sigma=sigma
        )

        assert eta_comp < eta_diff, (
            f"composite_gbs eta ({eta_comp:.3e}) should be < diffusion eta ({eta_diff:.3e})"
        )

    def test_diffusion_mode_ignores_sigma(self):
        """sigma parameter has no effect in diffusion mode."""
        T = 250.0
        d = 1.0e-3

        eta_no_sigma = IcePhysics.composite_viscosity(T, d_grain=d, creep_model="diffusion")
        eta_with_sigma = IcePhysics.composite_viscosity(
            T, d_grain=d, creep_model="diffusion", sigma=1.0e5
        )

        np.testing.assert_allclose(eta_no_sigma, eta_with_sigma, rtol=1e-15)

    def test_backward_compatibility_default_mode(self):
        """Default creep_model=None gives same result as explicit 'diffusion'."""
        T = 250.0
        d = 1.0e-3

        eta_default = IcePhysics.composite_viscosity(T, d_grain=d)
        eta_explicit = IcePhysics.composite_viscosity(
            T, d_grain=d, creep_model="diffusion"
        )

        np.testing.assert_allclose(eta_default, eta_explicit, rtol=1e-15)

    def test_composite_gbs_requires_sigma(self):
        """composite_gbs mode raises ValueError when sigma is None or <= 0."""
        T = 250.0
        with pytest.raises(ValueError, match="sigma"):
            IcePhysics.composite_viscosity(T, creep_model="composite_gbs")
        with pytest.raises(ValueError, match="sigma"):
            IcePhysics.composite_viscosity(T, creep_model="composite_gbs", sigma=0.0)
        with pytest.raises(ValueError, match="sigma"):
            IcePhysics.composite_viscosity(T, creep_model="composite_gbs", sigma=-100.0)

    def test_unknown_creep_model_raises(self):
        """Unknown creep_model string raises ValueError."""
        with pytest.raises(ValueError, match="Unknown creep_model"):
            IcePhysics.composite_viscosity(250.0, creep_model="bogus")


# =========================================================================
# Convective stress
# =========================================================================

class TestConvectiveStress:
    """Tests for IcePhysics.convective_stress."""

    def test_stress_order_of_magnitude(self):
        """Convective stress should be O(1-100 kPa) for Europa-like conditions.

        T_i=250K, D_conv=10km, Ra_i=1e6 => sigma ~ 3.8 kPa
        """
        sigma = IcePhysics.convective_stress(
            T_i=250.0,
            D_conv=10_000.0,
            Ra_i=1.0e6,
            Ra_crit=1000.0,
        )
        # Should be in 1-100 kPa range
        assert 1.0e3 < sigma < 1.0e5, (
            f"Convective stress {sigma:.1f} Pa not in O(1-100 kPa)"
        )
        # Check precise value against hand calculation
        np.testing.assert_allclose(sigma, 3793.0, rtol=1e-2)

    def test_stress_scales_with_D_conv(self):
        """Thicker convecting layer should produce larger stress (via delta_rh)."""
        sigma_thin = IcePhysics.convective_stress(250.0, 5_000.0, 1.0e6)
        sigma_thick = IcePhysics.convective_stress(250.0, 20_000.0, 1.0e6)
        assert sigma_thick > sigma_thin

    def test_stress_decreases_with_higher_Ra(self):
        """Higher Ra => thinner boundary layer => smaller delta_rh => smaller stress."""
        sigma_low_ra = IcePhysics.convective_stress(250.0, 10_000.0, 1.0e5)
        sigma_high_ra = IcePhysics.convective_stress(250.0, 10_000.0, 1.0e8)
        assert sigma_high_ra < sigma_low_ra
