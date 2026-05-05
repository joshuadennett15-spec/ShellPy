"""
Tests for Behn et al. (2021) ice paleowattmeter.

Covers:
    - equilibrium_grain_size physical range and stress dependence
    - grain_growth_rate positivity and monotonic decrease with d
    - grain_reduction_rate positivity and monotonic increase with d
    - solve_wattmeter convergence and growth-reduction balance
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from wattmeter import (
    equilibrium_grain_size, solve_wattmeter,
    grain_growth_rate, grain_reduction_rate,
    Q_GG, GAMMA_GB, C_GEOM, LAMBDA_DEFAULT, D_MIN, D_MAX,
)


# =========================================================================
# 1. Equilibrium grain size in physical range
# =========================================================================

class TestEquilibriumGrainSize:
    """Tests for equilibrium_grain_size."""

    def test_physical_range(self):
        """d_ss should be 0.01 -- 100 mm for typical convective conditions."""
        T = 260.0        # K  (warm convecting ice)
        sigma = 1.0e4    # Pa (10 kPa)
        eps_dot = 1.0e-13  # s^-1
        d = equilibrium_grain_size(T, sigma, eps_dot)
        assert 1e-5 < d < 0.1, f"d_ss={d*1e3:.3f} mm outside 0.01-100 mm"

    def test_higher_stress_smaller_grain(self):
        """Higher stress should produce smaller equilibrium grain size."""
        T = 260.0
        eps_dot = 1e-13
        d_lo = equilibrium_grain_size(T, sigma=1e3, eps_dot=eps_dot)
        d_hi = equilibrium_grain_size(T, sigma=1e5, eps_dot=eps_dot)
        assert d_hi < d_lo, "Higher stress must reduce d_ss"

    def test_higher_strain_rate_smaller_grain(self):
        """Higher strain rate should produce smaller grain size."""
        T = 260.0
        sigma = 1e4
        d_slow = equilibrium_grain_size(T, sigma, eps_dot=1e-14)
        d_fast = equilibrium_grain_size(T, sigma, eps_dot=1e-12)
        assert d_fast < d_slow, "Higher eps_dot must reduce d_ss"

    def test_higher_temp_larger_grain(self):
        """Higher T enhances growth more than reduction -> larger d_ss."""
        sigma = 1e4
        eps_dot = 1e-13
        d_cold = equilibrium_grain_size(240.0, sigma, eps_dot)
        d_warm = equilibrium_grain_size(268.0, sigma, eps_dot)
        assert d_warm > d_cold, "Warmer T must increase d_ss"

    def test_clipped_to_bounds(self):
        """Extreme inputs should still return within [D_MIN, D_MAX]."""
        # Very high stress + strain rate -> tiny grain -> clipped to D_MIN
        d = equilibrium_grain_size(200.0, 1e8, 1e-6)
        assert d >= D_MIN

        # Very low stress + strain rate -> huge grain -> clipped to D_MAX
        d = equilibrium_grain_size(270.0, 1.0, 1e-20)
        assert d <= D_MAX

    def test_bubble_rich_p6(self):
        """p=6.03 (bubble-rich) should also produce a physical grain size."""
        d = equilibrium_grain_size(260.0, 1e4, 1e-13, p=6.03)
        assert D_MIN <= d <= D_MAX


# =========================================================================
# 2. Growth rate
# =========================================================================

class TestGrainGrowthRate:
    """Tests for grain_growth_rate."""

    def test_positive(self):
        """Growth rate should always be positive."""
        rate = grain_growth_rate(T=260.0, d=1e-3)
        assert rate > 0

    def test_decreases_with_d(self):
        """Growth rate decreases as grain gets larger (1/d^(p-1) term)."""
        r_small = grain_growth_rate(T=260.0, d=1e-4)
        r_large = grain_growth_rate(T=260.0, d=1e-2)
        assert r_large < r_small, "Growth rate must decrease with d"


# =========================================================================
# 3. Reduction rate
# =========================================================================

class TestGrainReductionRate:
    """Tests for grain_reduction_rate."""

    def test_positive(self):
        """Reduction rate should be positive for positive inputs."""
        rate = grain_reduction_rate(d=1e-3, sigma=1e4, eps_dot=1e-13)
        assert rate > 0

    def test_increases_with_d(self):
        """Reduction rate increases with d (d^2 dependence)."""
        r_small = grain_reduction_rate(d=1e-4, sigma=1e4, eps_dot=1e-13)
        r_large = grain_reduction_rate(d=1e-2, sigma=1e4, eps_dot=1e-13)
        assert r_large > r_small, "Reduction rate must increase with d"


# =========================================================================
# 4. Self-consistent solver
# =========================================================================

class TestSolveWattmeter:
    """Tests for solve_wattmeter (Picard iteration)."""

    def test_converges(self):
        """solve_wattmeter should converge within 20 iterations."""
        # Typical Europa convecting layer conditions
        T_i = 260.0
        D_conv = 10_000.0   # 10 km
        Ra_i = 1.0e7
        d = solve_wattmeter(T_i, D_conv, Ra_i)
        assert D_MIN <= d <= D_MAX, f"d={d} outside physical bounds"

    def test_growth_equals_reduction_at_equilibrium(self):
        """At converged d, growth and reduction rates should match within 5%."""
        from Physics import IcePhysics

        T_i = 260.0
        D_conv = 10_000.0
        Ra_i = 1.0e7
        d = solve_wattmeter(T_i, D_conv, Ra_i)

        sigma = IcePhysics.convective_stress(T_i, D_conv, Ra_i)
        eps_gbs = float(IcePhysics.gbs_strain_rate(T_i, d, sigma))
        eps_dis = float(IcePhysics.dislocation_strain_rate(T_i, sigma))
        eps_dot = eps_gbs + eps_dis

        g = grain_growth_rate(T_i, d)
        r = grain_reduction_rate(d, sigma, eps_dot)
        rel_err = abs(g - r) / max(g, r)
        assert rel_err < 0.05, f"growth={g:.3e}, reduction={r:.3e}, err={rel_err:.1%}"

    def test_different_initial_guess(self):
        """Convergence should be insensitive to initial guess."""
        T_i = 260.0
        D_conv = 10_000.0
        Ra_i = 1.0e7
        d1 = solve_wattmeter(T_i, D_conv, Ra_i, d_guess=1e-4)
        d2 = solve_wattmeter(T_i, D_conv, Ra_i, d_guess=1e-2)
        assert abs(d1 - d2) / max(d1, d2) < 0.02, "Result should not depend on guess"

    def test_solve_wattmeter_raises_on_zero_strain(self):
        """Very low T produces eps_dot = 0 → should raise, not silently return."""
        with pytest.raises(RuntimeError, match="eps_dot.*<= 0"):
            solve_wattmeter(T_i=1.0, D_conv=10e3, Ra_i=5000.0)

    def test_solve_wattmeter_raises_on_max_iter(self):
        """With max_iter=1 and loose tolerance, should fail to converge."""
        with pytest.raises(RuntimeError, match="did not converge"):
            solve_wattmeter(T_i=260.0, D_conv=10e3, Ra_i=5000.0, max_iter=1, tol=1e-15)
