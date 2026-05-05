"""
Behn et al. (2021) Ice Paleowattmeter — Equilibrium Grain Size

Implements the steady-state grain size model from:
    Behn, Goldsby, Hirth (2021), "Grain-size evolution in subducting
    oceanic lithosphere", The Cryosphere, 15, 4589-4605.

Equilibrium is the balance between grain growth (normal grain growth
kinetics) and grain reduction (dynamic recrystallisation driven by
mechanical work).  The closed-form solution for d_ss avoids time
integration entirely.

Usage:
    d = equilibrium_grain_size(T, sigma, eps_dot)
    d = solve_wattmeter(T_i, D_conv, Ra_i)      # self-consistent loop
"""

import numpy as np
from Physics import IcePhysics

# =========================================================================
# Behn et al. (2021) Table 1 — Ice Ih parameters
# =========================================================================
R_GAS = 8.314          # J mol^-1 K^-1

Q_GG = 42.0e3          # J/mol  grain-growth activation energy
GAMMA_GB = 0.065       # J/m^2  grain-boundary energy
C_GEOM = 3.0           # —      geometric constant
LAMBDA_DEFAULT = 0.01  # —      work fraction to grain reduction

# Grain-growth rate constants K_gg (m^p / s)
K_GG_P2 = 1.0e-9      # bubble-free, p=2  (Azuma et al. 2012 estimate)
K_GG_P6 = 9.15e-18    # bubble-rich, p=6.03 (Behn 2021, joint fit)

# Physical bounds on grain size (m)
D_MIN = 1.0e-6         # 1 µm
D_MAX = 0.1            # 100 mm


def _k_gg_default(p: float) -> float:
    """Return default K_gg for a given growth exponent p."""
    if abs(p - 2.0) < 0.1:
        return K_GG_P2
    if abs(p - 6.03) < 0.1:
        return K_GG_P6
    raise ValueError(f"No default K_gg for p={p}; supply K_gg explicitly.")


def grain_growth_rate(T: float, d: float, p: float = 2.0,
                      K_gg: float = None) -> float:
    """
    Normal grain growth rate dd/dt (m/s).

    Behn (2021) Eq. 4:
        dd/dt = K_gg * exp(-Q_gg / (R*T)) / (p * d^(p-1))

    Args:
        T:    Temperature (K)
        d:    Current grain size (m)
        p:    Growth exponent (2.0 = bubble-free, 6.03 = bubble-rich)
        K_gg: Growth rate constant (m^p/s).  Default selected by p.

    Returns:
        Grain growth rate (m/s), always positive.
    """
    if K_gg is None:
        K_gg = _k_gg_default(p)
    return K_gg * np.exp(-Q_GG / (R_GAS * T)) / (p * d ** (p - 1))


def grain_reduction_rate(d: float, sigma: float, eps_dot: float,
                         lam: float = LAMBDA_DEFAULT) -> float:
    """
    Grain reduction rate |dd/dt| due to dynamic recrystallisation (m/s).

    Behn (2021) Eq. 12:
        |dd/dt| = lambda * d^2 * sigma * eps_dot / (c * gamma_gb)

    Args:
        d:        Current grain size (m)
        sigma:    Deviatoric stress (Pa)
        eps_dot:  Strain rate (s^-1)
        lam:      Fraction of mechanical work going to grain reduction

    Returns:
        Magnitude of grain reduction rate (m/s), always positive.
    """
    return lam * d ** 2 * sigma * eps_dot / (C_GEOM * GAMMA_GB)


def equilibrium_grain_size(T: float, sigma: float, eps_dot: float,
                           p: float = 2.0, K_gg: float = None,
                           lam: float = LAMBDA_DEFAULT) -> float:
    """
    Steady-state grain size d_ss (m) from the wattmeter balance.

    Setting growth = reduction and solving for d:
        d_ss = [K_gg * exp(-Q_gg/(R*T)) * c * gamma_gb
                / (p * lam * sigma * eps_dot)]^(1/(p+1))

    Clipped to [D_MIN, D_MAX] for physical plausibility.

    Args:
        T:        Temperature (K)
        sigma:    Deviatoric stress (Pa)
        eps_dot:  Strain rate (s^-1)
        p:        Growth exponent (default 2.0)
        K_gg:     Growth rate constant; default selected by p
        lam:      Work fraction (default 0.01)

    Returns:
        Equilibrium grain size (m).
    """
    if K_gg is None:
        K_gg = _k_gg_default(p)

    numerator = K_gg * np.exp(-Q_GG / (R_GAS * T)) * C_GEOM * GAMMA_GB
    denominator = p * lam * sigma * eps_dot

    d_ss = (numerator / denominator) ** (1.0 / (p + 1))
    return float(np.clip(d_ss, D_MIN, D_MAX))


def solve_wattmeter(T_i: float, D_conv: float, Ra_i: float,
                    d_guess: float = 1.0e-3, p: float = 2.0,
                    max_iter: int = 20, tol: float = 0.01) -> float:
    """
    Self-consistent wattmeter iteration: d -> sigma -> eps_dot -> d_eq.

    At each step:
        1. sigma = IcePhysics.convective_stress(T_i, D_conv, Ra_i)
        2. eps_dot = gbs(T_i, d, sigma) + dislocation(T_i, sigma)
        3. d_new = equilibrium_grain_size(T_i, sigma, eps_dot, p)

    Note: sigma depends on (T_i, D_conv, Ra_i) only, so it is constant
    across iterations.  The nonlinearity comes from eps_dot(d).

    Args:
        T_i:      Interior (convecting) temperature (K)
        D_conv:   Convective layer thickness (m)
        Ra_i:     Rayleigh number
        d_guess:  Initial grain size guess (m, default 1 mm)
        p:        Growth exponent (default 2.0)
        max_iter: Maximum Picard iterations (default 20)
        tol:      Relative convergence tolerance on d (default 1%)

    Returns:
        Converged equilibrium grain size (m).

    Raises:
        RuntimeError: If iteration does not converge within max_iter steps.
    """
    sigma = IcePhysics.convective_stress(T_i, D_conv, Ra_i)
    d = d_guess

    for iteration in range(max_iter):
        eps_gbs = float(IcePhysics.gbs_strain_rate(T_i, d, sigma))
        eps_dis = float(IcePhysics.dislocation_strain_rate(T_i, sigma))
        eps_dot = eps_gbs + eps_dis

        if eps_dot <= 0:
            raise RuntimeError(
                f"solve_wattmeter: eps_dot={eps_dot:.2e} <= 0 at T_i={T_i:.1f} K, "
                f"d={d:.2e} m, sigma={sigma:.2e} Pa (iteration {iteration}). "
                f"Strain rate underflow — temperature may be too low for creep."
            )

        d_new = equilibrium_grain_size(T_i, sigma, eps_dot, p=p)

        if abs(d_new - d) / max(d, D_MIN) < tol:
            return d_new

        d = d_new

    raise RuntimeError(
        f"solve_wattmeter: did not converge within {max_iter} iterations. "
        f"Last d={d:.2e} m, d_new={d_new:.2e} m, T_i={T_i:.1f} K."
    )
