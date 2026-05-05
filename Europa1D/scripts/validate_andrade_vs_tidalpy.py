#!/usr/bin/env python
"""
Validate our Andrade implementation against TidalPy's formulation.

TidalPy (Renaud 2023) uses:
    andrade_term = J_elastic * eta * omega * zeta
    const_term   = J_elastic * andrade_term^(-alpha) * Gamma(alpha)

Our code (McCarthy et al. 2011 convention) uses:
    andrade_term = omega * tau * zeta     (where tau = eta/mu)
    const_term   = J_elastic * andrade_term^(-alpha) * Gamma(1 + alpha)

Key differences:
    1. Gamma(alpha) vs Gamma(1+alpha):
       Since Gamma(1+alpha) = alpha * Gamma(alpha), this is a factor of alpha.
       TidalPy uses Gamma(alpha), we use Gamma(1+alpha).
       For alpha=0.2: Gamma(0.2) = 4.5908, Gamma(1.2) = 0.9182.
       Our const_term is smaller by a factor of alpha = 0.2.

    2. TidalPy default alpha=0.3, ours alpha=0.2.
       Both are literature values; 0.2 is McCarthy et al. 2011 for ice,
       0.3 is more common for silicates.

    3. The andrade_term formulation is equivalent:
       TidalPy: J_elastic * eta * omega * zeta = (1/mu) * eta * omega * zeta
                = omega * (eta/mu) * zeta = omega * tau * zeta  (same as ours)

This script compares dissipation rates across temperature for both
conventions to quantify the difference.
"""
import sys
import os
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from math import gamma

# Europa parameters
omega = 2.047e-5       # rad/s (orbital frequency)
mu = 3.3e9             # Pa (shear modulus)
epsilon_0 = 1.0e-5     # tidal strain
J_elastic = 1.0 / mu
zeta = 1.0

# Temperature range (warm convecting ice)
T_range = np.linspace(200, 270, 8)  # K
Q_v = 59400.0          # J/mol
R = 8.314              # J/(mol*K)
eta_ref = 5e13         # Pa*s (Green et al. 2021)
T_melt = 273.0


def viscosity(T):
    """Simple Arrhenius viscosity."""
    return eta_ref * np.exp((Q_v / R) * (1.0 / T - 1.0 / T_melt))


def andrade_ours(T, alpha=0.2):
    """Our implementation (McCarthy et al. 2011 convention)."""
    eta = viscosity(T)
    tau = eta / mu

    andrade_term = np.clip(omega * tau * zeta, 1e-100, None)
    const_term = J_elastic * (andrade_term ** -alpha) * gamma(1 + alpha)

    J_real = J_elastic + const_term * np.cos(alpha * np.pi / 2.0)
    J_imag = J_elastic / (omega * tau) + const_term * np.sin(alpha * np.pi / 2.0)

    G_imag = J_imag / (J_real**2 + J_imag**2)
    return 0.5 * omega * epsilon_0**2 * G_imag


def andrade_tidalpy(T, alpha=0.3):
    """TidalPy convention (Renaud 2023): Gamma(alpha), default alpha=0.3."""
    eta = viscosity(T)
    tau = eta / mu

    andrade_term = np.clip(omega * tau * zeta, 1e-100, None)
    # TidalPy uses Gamma(alpha), not Gamma(1+alpha)
    const_term = J_elastic * (andrade_term ** -alpha) * gamma(alpha)

    # TidalPy builds complex compliance then extracts Im(1/J*)
    J_real = J_elastic + const_term * np.cos(alpha * np.pi / 2.0)
    J_imag = J_elastic / (omega * tau) + const_term * np.sin(alpha * np.pi / 2.0)

    G_imag = J_imag / (J_real**2 + J_imag**2)
    return 0.5 * omega * epsilon_0**2 * G_imag


def maxwell(T):
    """Maxwell viscoelastic (for reference)."""
    eta = viscosity(T)
    num = epsilon_0**2 * omega**2 * eta
    den = 2.0 * (1.0 + (omega * eta / mu) ** 2)
    return num / den


print("=" * 90)
print("Andrade Implementation Comparison: Ours vs TidalPy Convention")
print("=" * 90)
print()
print(f"Parameters: omega={omega:.3e} rad/s, mu={mu:.1e} Pa, eps0={epsilon_0:.1e}")
print(f"            eta_ref={eta_ref:.1e} Pa*s, Q_v={Q_v:.0f} J/mol")
print()

# Header
print(f"{'T (K)':>7} | {'eta (Pa*s)':>12} | {'omega*tau':>10} | "
      f"{'Maxwell':>12} | {'Ours (a=0.2)':>14} | {'TidalPy (a=0.3)':>16} | "
      f"{'Ours/Max':>9} | {'TP/Max':>8}")
print("-" * 110)

for T in T_range:
    eta = viscosity(T)
    tau = eta / mu
    wt = omega * tau

    q_max = maxwell(T)
    q_ours = andrade_ours(T, alpha=0.2)
    q_tp = andrade_tidalpy(T, alpha=0.3)

    ratio_ours = q_ours / q_max if q_max > 0 else 0
    ratio_tp = q_tp / q_max if q_max > 0 else 0

    print(f"{T:7.1f} | {eta:12.3e} | {wt:10.3e} | "
          f"{q_max:12.4e} | {q_ours:14.4e} | {q_tp:16.4e} | "
          f"{ratio_ours:9.2f} | {ratio_tp:8.2f}")

print()
print("=" * 90)
print("Interpretation:")
print("  Ours/Max > 1  means Andrade produces MORE heating than Maxwell at that T.")
print("  The Andrade enhancement is strongest far from the Maxwell peak (omega*tau >> 1")
print("  or omega*tau << 1), where Maxwell gives almost zero dissipation but Andrade")
print("  still dissipates via transient creep.")
print()

# Same-alpha comparison to isolate the Gamma function difference
print("=" * 90)
print("Same-alpha comparison (both at alpha=0.2) to isolate Gamma convention:")
print("=" * 90)
print()
print(f"{'T (K)':>7} | {'Ours Gamma(1+a)':>16} | {'TidalPy Gamma(a)':>17} | {'Ratio (TP/Ours)':>16}")
print("-" * 65)

for T in T_range:
    q_ours = andrade_ours(T, alpha=0.2)
    q_tp_same = andrade_tidalpy(T, alpha=0.2)  # Same alpha, different Gamma
    ratio = q_tp_same / q_ours if q_ours > 0 else 0
    print(f"{T:7.1f} | {q_ours:16.4e} | {q_tp_same:17.4e} | {ratio:16.3f}")

print()
print(f"Gamma(0.2) = {gamma(0.2):.4f}")
print(f"Gamma(1.2) = {gamma(1.2):.4f}")
print(f"Gamma(0.2) / Gamma(1.2) = {gamma(0.2)/gamma(1.2):.4f}  (= 1/alpha = {1/0.2:.1f})")
print()
print("The TidalPy convention (Gamma(alpha)) produces a const_term that is")
print("1/alpha = 5x larger than ours (Gamma(1+alpha)). This amplifies the")
print("Andrade contribution but does NOT change the physics if the alpha")
print("parameter is calibrated consistently with the convention used.")
print()
print("McCarthy et al. (2011) Eq. 1 uses Gamma(1+alpha) -- our convention is correct")
print("for the McCarthy formulation. TidalPy follows Efroimsky (2012) which absorbs")
print("the alpha factor differently. Both are valid; they just define alpha differently.")
