"""
Validation tests for Andrade tidal dissipation against Renaud & Henning (2018).

Reference: Renaud & Henning (2018), ApJ 857:98, Table 1.
Parameters: alpha=0.2, zeta=1.0.

These tests validate our IcePhysics.tidal_heating() implementation by computing
an independent reference from the published Andrade compliance equations and
checking agreement within 1%.
"""

import numpy as np
import pytest
import sys
import os
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from Physics import IcePhysics
from constants import Planetary, Rheology, Thermal


# ---------------------------------------------------------------------------
# Helpers: independent reference calculation
# ---------------------------------------------------------------------------

def _frank_kamenetskii_viscosity(T, eta_ref, T_melt, Q_v, R):
    """Reproduce the FK viscosity outside IcePhysics to keep reference independent."""
    return eta_ref * np.exp((Q_v / R) * (1.0 / T - 1.0 / T_melt))


def _andrade_heating_reference(T, eta, mu, epsilon_0, omega, alpha, zeta):
    """Independent Andrade dissipation from Renaud & Henning (2018) Eq. in Table 2.

    Complex compliance:
        J*(w) = J_u - i/(eta*w) + J_u*(i*J_u*eta*zeta*w)^(-alpha) * Gamma(1+alpha)

    Volumetric heating:
        q = 0.5 * omega * epsilon_0^2 * Im(G*)
    where Im(G*) = J_imag / (J_real^2 + J_imag^2).
    """
    from scipy.special import gamma as gamma_func

    J_u = 1.0 / mu
    tau = eta / mu
    andrade_term = omega * tau * zeta

    const = J_u * andrade_term ** (-alpha) * gamma_func(1 + alpha)
    J_real = J_u + const * np.cos(alpha * np.pi / 2.0)
    J_imag = J_u / (omega * tau) + const * np.sin(alpha * np.pi / 2.0)

    G_imag = J_imag / (J_real ** 2 + J_imag ** 2)
    return 0.5 * omega * epsilon_0 ** 2 * G_imag


def _maxwell_heating_reference(eta, mu, epsilon_0, omega):
    """Independent Maxwell dissipation for cross-check."""
    numerator = epsilon_0 ** 2 * omega ** 2 * eta
    denominator = 2.0 * (1.0 + (omega ** 2 * eta ** 2 / mu ** 2))
    return numerator / denominator


# ---------------------------------------------------------------------------
# Test: Andrade dissipation matches independent calculation within 1%
# ---------------------------------------------------------------------------

class TestAndradeValidation:
    """Validate tidal_heating Andrade path against Renaud & Henning (2018)."""

    # Common Europa-like parameters
    T = 250.0
    ALPHA = 0.2
    ZETA = 1.0
    MU = Rheology.RIGIDITY_ICE          # 3.3e9 Pa
    EPSILON_0 = 1.0e-5                  # tidal strain
    OMEGA = Planetary.ORBITAL_FREQ      # 2.047e-5 rad/s
    ETA_REF = Rheology.VISCOSITY_REF    # 5e13 Pa s
    T_MELT = Thermal.MELT_TEMP         # 273 K

    @pytest.fixture(autouse=True)
    def _compute_reference(self):
        """Pre-compute the independent reference values used by all tests."""
        self.eta = _frank_kamenetskii_viscosity(
            self.T, self.ETA_REF, self.T_MELT,
            Rheology.ACTIVATION_ENERGY_V, Rheology.GAS_CONSTANT
        )
        self.q_andrade_ref = _andrade_heating_reference(
            self.T, self.eta, self.MU, self.EPSILON_0,
            self.OMEGA, self.ALPHA, self.ZETA
        )
        self.q_maxwell_ref = _maxwell_heating_reference(
            self.eta, self.MU, self.EPSILON_0, self.OMEGA
        )

    @patch.object(Rheology, 'MODEL', 'Andrade')
    def test_andrade_dissipation_matches_reference(self):
        """Our Andrade tidal heating must match independent calc within 1%.

        Reference: Renaud & Henning (2018), ApJ 857:98, Table 1.
        Parameters: alpha=0.2, zeta=1.0.
        """
        q_ours = IcePhysics.tidal_heating(
            T=self.T,
            epsilon_0=self.EPSILON_0,
            mu_ice=self.MU,
            eta_ref=self.ETA_REF,
        )

        assert q_ours == pytest.approx(self.q_andrade_ref, rel=0.01), (
            f"Our: {q_ours:.6e}, Reference: {self.q_andrade_ref:.6e}"
        )

    @patch.object(Rheology, 'MODEL', 'Maxwell')
    def test_maxwell_dissipation_matches_reference(self):
        """Our Maxwell tidal heating must match independent calc within 1%."""
        q_ours = IcePhysics.tidal_heating(
            T=self.T,
            epsilon_0=self.EPSILON_0,
            mu_ice=self.MU,
            eta_ref=self.ETA_REF,
        )

        assert q_ours == pytest.approx(self.q_maxwell_ref, rel=0.01), (
            f"Our: {q_ours:.6e}, Reference: {self.q_maxwell_ref:.6e}"
        )

    def test_andrade_greater_than_maxwell_high_viscosity(self):
        """Andrade exceeds Maxwell at high omega*tau (Renaud & Henning 2018).

        At high omega*tau (stiff/cold ice), the Andrade transient creep term
        provides significantly more dissipation than the Maxwell model alone.
        This is because the Andrade J_imag contribution decays as (omega*tau)^{-alpha}
        (slow power-law) while Maxwell J_imag decays as (omega*tau)^{-1} (fast).

        We use eta_ref=1e16 Pa s to reach omega*tau ~ 62 where the relationship
        is unambiguous.
        """
        eta_ref_high = 1.0e16  # high viscosity => omega*tau >> 1

        with patch.object(Rheology, 'MODEL', 'Andrade'):
            q_andrade = IcePhysics.tidal_heating(
                T=self.T_MELT,   # use T_melt so viscosity = eta_ref exactly
                epsilon_0=self.EPSILON_0,
                mu_ice=self.MU,
                eta_ref=eta_ref_high,
            )

        with patch.object(Rheology, 'MODEL', 'Maxwell'):
            q_maxwell = IcePhysics.tidal_heating(
                T=self.T_MELT,
                epsilon_0=self.EPSILON_0,
                mu_ice=self.MU,
                eta_ref=eta_ref_high,
            )

        assert q_andrade > q_maxwell, (
            f"Andrade ({q_andrade:.4e}) should exceed Maxwell ({q_maxwell:.4e}) "
            f"at high omega*tau"
        )

    def test_andrade_less_than_maxwell_near_peak(self):
        """Near the Maxwell peak (omega*tau ~ 1-5), Andrade can be less than Maxwell.

        The Andrade term adds to both J_real and J_imag, but the J_real increase
        inflates |J*|^2 faster than J_imag grows, reducing G_imag = J_imag/|J*|^2.
        This is correct physics, not a bug.
        """
        with patch.object(Rheology, 'MODEL', 'Andrade'):
            q_andrade = IcePhysics.tidal_heating(
                T=self.T,
                epsilon_0=self.EPSILON_0,
                mu_ice=self.MU,
                eta_ref=self.ETA_REF,
            )

        with patch.object(Rheology, 'MODEL', 'Maxwell'):
            q_maxwell = IcePhysics.tidal_heating(
                T=self.T,
                epsilon_0=self.EPSILON_0,
                mu_ice=self.MU,
                eta_ref=self.ETA_REF,
            )

        # At T=250K with default eta_ref=5e13, omega*tau ~ 3.4
        # Andrade < Maxwell in this regime
        assert q_andrade < q_maxwell, (
            f"Near Maxwell peak: Andrade ({q_andrade:.4e}) should be less than "
            f"Maxwell ({q_maxwell:.4e})"
        )


# ---------------------------------------------------------------------------
# Test: sensitivity to alpha and zeta
# ---------------------------------------------------------------------------

class TestAndradeSensitivity:
    """Verify Andrade heating responds correctly to rheological parameters."""

    T = 250.0
    MU = Rheology.RIGIDITY_ICE
    EPSILON_0 = 1.0e-5
    OMEGA = Planetary.ORBITAL_FREQ

    @patch.object(Rheology, 'MODEL', 'Andrade')
    def test_higher_alpha_increases_dissipation(self):
        """Increasing Andrade alpha (more transient creep) increases dissipation.

        At Europa's forcing frequency, higher alpha strengthens the Andrade
        contribution to Im(J*), increasing volumetric heating.
        """
        results = []
        for alpha_val in [0.15, 0.2, 0.3]:
            with patch.object(Rheology, 'ANDRADE_ALPHA', alpha_val):
                q = IcePhysics.tidal_heating(
                    T=self.T,
                    epsilon_0=self.EPSILON_0,
                    mu_ice=self.MU,
                )
            results.append(q)

        # Each should be strictly increasing
        assert results[0] < results[1] < results[2], (
            f"Expected increasing q with alpha: {results}"
        )

    @patch.object(Rheology, 'MODEL', 'Andrade')
    def test_heating_is_finite_and_positive(self):
        """Tidal heating must be finite and positive for physical parameters."""
        q = IcePhysics.tidal_heating(
            T=self.T,
            epsilon_0=self.EPSILON_0,
            mu_ice=self.MU,
        )
        assert np.isfinite(q)
        assert q > 0


# ---------------------------------------------------------------------------
# Test: TidalPy cross-validation (skipped if not installed)
# ---------------------------------------------------------------------------

def _tidalpy_available():
    """Check if TidalPy is importable."""
    try:
        import TidalPy  # noqa: F401
        return True
    except ImportError:
        return False


class TestTidalPyCrossValidation:
    """Cross-validate against TidalPy if available."""

    @pytest.mark.skipif(
        not _tidalpy_available(),
        reason="TidalPy not installed"
    )
    def test_placeholder(self):
        """Placeholder for future TidalPy comparison."""
        pytest.skip("TidalPy integration test not yet implemented")
