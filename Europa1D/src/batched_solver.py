"""
Batched Numba 1D Ice Shell Solver.

Green-path parity port of Solver.py for latitude-batched execution.
See docs/superpowers/specs/2026-03-22-batched-numba-solver-design.md
"""
import numpy as np
from numba import njit


@njit
def _conductivity(T):
    """Carnahan et al. (2021): k(T) = 612 / T  [W/m·K]"""
    return 612.0 / T


@njit
def _specific_heat(T):
    """cp(T) = 7.49*T + 90  [J/kg·K]"""
    return 7.49 * T + 90.0


@njit
def _density_ice(T):
    """rho(T) = 917 * (1 + 1.6e-4 * (273 - T))  [kg/m³]"""
    return 917.0 * (1.0 + 1.6e-4 * (273.0 - T))


@njit
def _basal_melting_point(H, g):
    """Pressure-dependent melting: T_m = 273 + CC * rho * g * H  [K]"""
    return 273.0 + (-7.4e-8) * 917.0 * g * H


@njit
def _effective_conductivity(T, porosity, salt_frac, salt_scale, por_cure_temp):
    """k_eff with porosity and salt corrections."""
    k = 612.0 / T
    if porosity > 0.0 and T < por_cure_temp:
        k = k * (1.0 - porosity)
    if salt_frac > 0.0:
        k = k * (1.0 + salt_frac * (salt_scale - 1.0))
    return k


# =============================================================================
# 2. VISCOSITY HELPERS
# =============================================================================

@njit
def _viscosity_simple(T, eta_ref, Q_v, R, T_melt):
    """Frank-Kamenetskii: eta = eta_ref * exp((Q_v/R)*(1/T - 1/T_melt))"""
    T_safe = max(T, 50.0)
    return eta_ref * np.exp((Q_v / R) * (1.0 / T_safe - 1.0 / T_melt))


@njit
def _composite_viscosity(T, d_grain, d_del, D0v, D0b, Q_v, Q_b, d_molar, R,
                         p_grain=2.0, d_ref=1.0e-3):
    """Grain-size-dependent viscosity with reference-grain normalization.

    p_grain=2.0 recovers Nabarro-Herring (Howell 2021).
    p_grain=1.4 follows GBS (Goldsby & Kohlstedt 2001).
    p_grain=1.1 follows GSS (Prior et al. 2025).
    """
    T_safe = max(T, 50.0)
    Dv = D0v * np.exp(-Q_v / (R * T_safe))
    Db = D0b * np.exp(-Q_b / (R * T_safe))
    # Reference-grain normalization: matches N-H at d=d_ref, scales as d^p
    d_eff = (d_grain / d_ref) ** p_grain * d_ref * d_ref
    prefactor = (42.0 * d_molar) / (R * T_safe * d_eff)
    diff_term = Dv + (np.pi * d_del / d_grain) * Db
    eta = 0.5 / (prefactor * diff_term)
    if eta < 1e12:
        return 1e12
    if eta > 1e25:
        return 1e25
    return eta


# =============================================================================
# 3. TIDAL HEATING HELPERS
# =============================================================================

@njit
def _tidal_heating_maxwell(eps0, omega, eta, mu):
    """Maxwell viscoelastic dissipation [W/m³]."""
    num = eps0 * eps0 * omega * omega * eta
    den = 2.0 * (1.0 + (omega * omega * eta * eta / (mu * mu)))
    return num / den


@njit
def _tidal_heating_andrade(eps0, omega, eta, mu, alpha, zeta, gamma_val):
    """Andrade transient creep dissipation [W/m³]."""
    J_elastic = 1.0 / mu
    tau = eta / mu
    andrade_term = omega * tau * zeta
    if andrade_term < 1e-100:
        andrade_term = 1e-100
    const_term = J_elastic * (andrade_term ** (-alpha)) * gamma_val
    J_real = J_elastic + const_term * np.cos(alpha * np.pi / 2.0)
    J_imag = J_elastic / (omega * tau) + const_term * np.sin(alpha * np.pi / 2.0)
    G_imag = J_imag / (J_real * J_real + J_imag * J_imag)
    return 0.5 * omega * eps0 * eps0 * G_imag


# =============================================================================
# 4. CONVECTION HELPERS
# =============================================================================

@njit
def _howell_Tc(T_melt, Q_v, R):
    """Howell (2021) conductive-base temperature."""
    if not np.isfinite(T_melt) or not np.isfinite(Q_v) or Q_v <= 0.0:
        safe_melt = 273.0 if not np.isfinite(T_melt) else max(T_melt, 1.0)
        return 0.88 * safe_melt
    ratio = R / Q_v
    if not np.isfinite(ratio) or abs(ratio) < 1e-30:
        return 0.88 * max(T_melt, 1.0)
    radicand = 4.0 * T_melt * ratio + 1.0
    if not np.isfinite(radicand) or radicand <= 0.0:
        return 0.88 * max(T_melt, 1.0)
    T_c = (np.sqrt(radicand) - 1.0) / (2.0 * ratio)
    if not np.isfinite(T_c):
        return 0.88 * max(T_melt, 1.0)
    return 2.0 * T_c - T_melt


@njit
def _deschamps_Ti(T_melt, T_surf, Q_v, R, c1, c2):
    """Deschamps & Vilella (2021) Eq. 18: interior temperature."""
    if not np.isfinite(T_melt) or not np.isfinite(T_surf) or not np.isfinite(Q_v) or Q_v <= 0.0:
        safe_melt = 273.0 if not np.isfinite(T_melt) else T_melt
        safe_surf = 104.0 if not np.isfinite(T_surf) else T_surf
        return 0.5 * (safe_melt + safe_surf)
    DTs = T_melt - T_surf
    B = Q_v / (2.0 * R * c1)
    if not np.isfinite(B) or abs(B) < 1e-30:
        return 0.5 * (T_melt + T_surf)
    radicand = 1.0 + (2.0 / B) * (T_melt - c2 * DTs)
    if not np.isfinite(radicand) or radicand <= 0.0:
        return 0.5 * (T_melt + T_surf)
    Ti = B * (np.sqrt(radicand) - 1.0)
    if not np.isfinite(Ti):
        return 0.5 * (T_melt + T_surf)
    return Ti


@njit
def _green_Tc_Ti(T_melt, T_surf, Q_v, R, eta_ref, theta_lid,
                 use_composite_transition_closure=False,
                 d_grain=1.0e-3, d_del=7.13e-10, D0v=9.1e-4, D0b=8.4e-4,
                 Q_b=49000.0, d_molar=1.97e-5, p_grain=2.0):
    """Green et al. (2021) lid base temperature. Returns (Tc, Ti)."""
    if (not np.isfinite(T_melt) or not np.isfinite(T_surf)
            or not np.isfinite(Q_v) or Q_v <= 0.0
            or T_melt <= T_surf + 1e-6):
        safe_T = 0.5 * (
            (273.0 if not np.isfinite(T_melt) else T_melt) +
            (104.0 if not np.isfinite(T_surf) else T_surf)
        )
        return safe_T, safe_T
    Ti = _deschamps_Ti(T_melt, T_surf, Q_v, R, 1.43, -0.03)
    if Ti < T_surf + 1.0:
        Ti = T_surf + 1.0
    if Ti > T_melt - 1.0:
        Ti = T_melt - 1.0

    fallback_to_analytic = True
    DTv = 0.0

    if use_composite_transition_closure:
        dT = 0.1
        eta_plus = _composite_viscosity(
            Ti + dT, d_grain, d_del, D0v, D0b, Q_v, Q_b, d_molar, R, p_grain
        )
        eta_minus = _composite_viscosity(
            Ti - dT, d_grain, d_del, D0v, D0b, Q_v, Q_b, d_molar, R, p_grain
        )

        if eta_plus > 0.0 and eta_minus > 0.0:
            dlneta_dTi = (np.log(eta_plus) - np.log(eta_minus)) / (2.0 * dT)
            if np.isfinite(dlneta_dTi) and abs(dlneta_dTi) > 1e-10:
                DTv = -1.0 / dlneta_dTi
                fallback_to_analytic = False

    if fallback_to_analytic:
        A = Q_v / (R * T_melt)
        exponent = A * ((T_melt / Ti) - 1.0)
        if exponent > 500.0:
            exponent = 500.0
        if exponent < -500.0:
            exponent = -500.0

        exp_term = np.exp(exponent)
        dni_dTi = -eta_ref * (A * T_melt / (Ti * Ti)) * exp_term

        if abs(dni_dTi) < 1e-100 or not np.isfinite(dni_dTi):
            Tc = _howell_Tc(T_melt, Q_v, R)
            if Tc < T_surf + 1.0:
                Tc = T_surf + 1.0
            if Tc > T_melt - 1.0:
                Tc = T_melt - 1.0
            return Tc, Ti

        ni = eta_ref * exp_term
        DTv = -ni / dni_dTi

    DTe = theta_lid * DTv
    Tc = Ti - DTe

    if Tc < T_surf + 1.0:
        Tc = T_surf + 1.0
    if Tc > T_melt - 1.0:
        Tc = T_melt - 1.0
    return Tc, Ti


@njit
def _rayleigh_number(DT, d, T_mean, eta, g, alpha_exp):
    """Ra = rho*g*alpha*DT*d^3 / (kappa*eta)"""
    rho = _density_ice(T_mean)
    k = 612.0 / T_mean
    cp = _specific_heat(T_mean)
    kappa = k / (rho * cp)
    return rho * g * alpha_exp * DT * d * d * d / (kappa * eta)


@njit
def _nusselt_simple(Ra, Ra_crit, C, xi):
    """Solomatov & Moresi (2000): Nu = C * Ra^xi if Ra >= Ra_crit, else 1."""
    if Ra < Ra_crit:
        return 1.0
    Nu = C * (Ra ** xi)
    if Nu < 1.0:
        return 1.0
    return Nu


@njit
def _nusselt_green(Ra, Ti, Tc, DT, Ra_crit, C, xi, zeta):
    """Green et al. (2021) Nu with internal heating correction."""
    if Ra < Ra_crit or DT <= 0.0:
        return 1.0
    temp_ratio = (Ti - Tc) / DT
    if temp_ratio < 0.01:
        temp_ratio = 0.01
    Nu = C * (Ra ** xi) * (temp_ratio ** zeta)
    if Nu < 1.0:
        return 1.0
    return Nu


@njit
def _nusselt_isoviscous(Ra, Ra_crit):
    """Isoviscous scaling: Nu = 0.088 * Ra^0.28. Diagnostic only."""
    if Ra < Ra_crit:
        return 1.0
    Nu = 0.088 * (Ra ** 0.28)
    if Nu < 1.0:
        return 1.0
    return Nu


# Nu scaling IDs for Numba (strings not supported in @njit):
#   0 = "green"  (FK viscosity + Green Nu)
#   1 = "howell" (composite viscosity + simple Nu)
#   2 = "isoviscous_benchmark" (FK viscosity + isoviscous Nu)
#   3 = "dv2021" (Deschamps & Vilella 2021, FK viscosity)
NU_SCALING_GREEN = 0
NU_SCALING_HOWELL = 1
NU_SCALING_ISOVISCOUS = 2
NU_SCALING_DV2021 = 3


@njit
def _nusselt_dv2021(Ra, Tc, T_melt, Q_v, R_gas, Ra_crit,
                    fk_correction_factor=0.75):
    """
    Deschamps & Vilella (2021) Phi_top as effective Nusselt number.

    Simplified for H_tilde=0 (bottom-heated limit, Ur<1 regime).
    Uses FK viscosity at the lid-base (T_c) for Ra_surf, then:
      T_m_tilde = 1 - a1/(f^2 * gamma)  [Eq. 21, H=0]
      Ra_eff = Ra_surf * exp(gamma * T_m_tilde)
      Phi_top = a * Ra_eff^b / gamma^c  [Eq. 23]

    The Ra passed here is computed at T_mean; we recompute Ra_surf internally.
    """
    if Ra < Ra_crit:
        return 1.0

    DT = T_melt - Tc
    if DT <= 0.0 or Tc <= 0.0 or T_melt <= 0.0:
        return 1.0

    # gamma = Q_v/R * (1/Tc - 1/T_melt)  [FK viscosity contrast]
    gamma = Q_v / R_gas * (1.0 / Tc - 1.0 / T_melt)
    if gamma < 1.0:
        gamma = 1.0

    # For H_tilde=0, T_m_tilde is analytic (no brentq needed)
    f = 1.0  # Cartesian thin-shell limit
    a1 = 1.23
    T_m_tilde = 1.0 - a1 / (f * f * gamma)

    # Ra_eff = Ra_surf * exp(gamma * T_m_tilde)
    # Ra was computed at T_mean. Convert to Ra_surf (at Tc):
    # Ra_surf = Ra * eta(T_mean) / eta(Tc) = Ra * exp(Q/R*(1/T_mean - 1/Tc))
    T_mean = (T_melt + Tc) / 2.0
    Ra_surf = Ra * np.exp(Q_v / R_gas * (1.0 / T_mean - 1.0 / Tc))
    Ra_eff = Ra_surf * np.exp(gamma * T_m_tilde)

    # Ur<1 coefficients: a=1.46, b=0.27, c=1.21
    a_coeff = 1.46
    b_coeff = 0.27
    c_coeff = 1.21

    Phi_top = a_coeff * (Ra_eff ** b_coeff) / (gamma ** c_coeff)

    # FK correction factor
    Phi_top *= fk_correction_factor

    if Phi_top < 1.0:
        return 1.0
    return Phi_top


@njit
def _harmonic_mean(k, k_half, nz):
    """Fill k_half with harmonic mean half-node conductivities."""
    for j in range(nz - 1):
        k_half[j] = 2.0 * k[j] * k[j + 1] / (k[j] + k[j + 1] + 1e-30)


@njit
def _scan_profile(T_col, nz, H, T_melt_basal, T_surf,
                  Q_v, Q_b, d_grain, d_del, D0v, D0b, d_molar,
                  eta_ref, R_gas,
                  theta_lid, c1, c2,
                  Ra_crit, Nu_C, Nu_xi, Nu_zeta,
                  alpha_exp, g,
                  use_composite_transition_closure=False,
                  p_grain=2.0,
                  nu_scaling_id=0):
    """Phase 2 profile scan. Returns 8 scalars matching ConvectionState fields.

    nu_scaling_id: 0=green (FK visc), 1=howell (composite visc), 2=isoviscous (FK visc)

    Returns: (idx_c, z_c, D_cond, D_conv, Tc, Ra, Nu, is_convecting)
    """
    if nz <= 0 or H <= 0.0 or not np.isfinite(T_melt_basal) or T_melt_basal <= T_surf + 1e-6:
        return max(nz - 1, 0), H, H, 0.0, T_surf, 0.0, 1.0, False

    Tc, Ti = _green_Tc_Ti(
        T_melt_basal, T_surf, Q_v, R_gas, eta_ref, theta_lid,
        use_composite_transition_closure, d_grain, d_del, D0v, D0b,
        Q_b, d_molar, p_grain
    )

    # Scan for first warm index
    idx_c = nz - 1
    found = False
    for i in range(nz):
        if T_col[i] >= Tc:
            idx_c = i
            found = True
            break

    if not found:
        return idx_c, H, H, 0.0, Tc, 0.0, 1.0, False

    # Interpolate z_c
    if 0 < idx_c < nz:
        T_above = T_col[idx_c - 1]
        T_below = T_col[idx_c]
        z_above = (idx_c - 1) / (nz - 1) * H
        z_below = idx_c / (nz - 1) * H
        if T_below > T_above:
            frac = (Tc - T_above) / (T_below - T_above)
            z_c = z_above + frac * (z_below - z_above)
        else:
            z_c = z_below
    else:
        z_c = idx_c / (nz - 1) * H

    D_cond = z_c
    D_conv = H - z_c
    if D_conv <= 0.0:
        return idx_c, z_c, D_cond, 0.0, Tc, 0.0, 1.0, False

    # Ra — viscosity model must match the Nu scaling law's calibration basis
    DT = T_melt_basal - Tc
    T_mean = (T_melt_basal + Tc) / 2.0
    if nu_scaling_id == NU_SCALING_HOWELL:
        # Composite (grain-size) viscosity — matches Howell simple Nu
        eta_mean = _composite_viscosity(T_mean, d_grain, d_del,
                                         D0v, D0b, Q_v, Q_b,
                                         d_molar, R_gas, p_grain)
    else:
        # FK viscosity — matches Green, isoviscous, and DV2021 calibration basis
        eta_mean = _viscosity_simple(T_mean, eta_ref, Q_v, R_gas, T_melt_basal)
    Ra = _rayleigh_number(DT, D_conv, T_mean, eta_mean, g, alpha_exp)

    # Nu from matched scaling law
    if nu_scaling_id == NU_SCALING_GREEN:
        Nu = _nusselt_green(Ra, Ti, Tc, DT, Ra_crit, Nu_C, Nu_xi, Nu_zeta)
    elif nu_scaling_id == NU_SCALING_ISOVISCOUS:
        Nu = _nusselt_isoviscous(Ra, Ra_crit)
    elif nu_scaling_id == NU_SCALING_DV2021:
        Nu = _nusselt_dv2021(Ra, Tc, T_melt_basal, Q_v, R_gas, Ra_crit)
    else:
        Nu = _nusselt_simple(Ra, Ra_crit, Nu_C, Nu_xi)

    is_convecting = Ra >= Ra_crit
    return idx_c, z_c, D_cond, D_conv, Tc, Ra, Nu, is_convecting


@njit
def _build_k_profile(T_col, k_impl, nz, H, T_melt_basal, T_surf,
                     Q_v, Q_b, d_grain, d_del, D0v, D0b, d_molar,
                     eta_ref, R_gas, theta_lid, c1, c2,
                     Ra_crit, Nu_C, Nu_xi, Nu_zeta, alpha_exp, g,
                     porosity, salt_frac, salt_scale, por_cure_temp,
                     nu_ramp,
                     use_composite_transition_closure=False,
                     p_grain=2.0,
                     nu_scaling_id=0):
    """Build conductivity profile with convection enhancement.
    Fills k_impl in-place. Returns convection diagnostics as 8-tuple."""
    # Step 1: scan for convection interface
    idx_c, z_c, D_cond, D_conv, Tc, Ra, Nu, is_conv = _scan_profile(
        T_col, nz, H, T_melt_basal, T_surf,
        Q_v, Q_b, d_grain, d_del, D0v, D0b, d_molar,
        eta_ref, R_gas, theta_lid, c1, c2,
        Ra_crit, Nu_C, Nu_xi, Nu_zeta, alpha_exp, g,
        use_composite_transition_closure, p_grain, nu_scaling_id)

    # Step 2: base conductivity + porosity/salt
    for i in range(nz):
        k_impl[i] = _effective_conductivity(T_col[i], porosity, salt_frac,
                                             salt_scale, por_cure_temp)

    # Step 3: Nu enhancement below z_c
    if idx_c < nz:
        Nu_eff = 1.0 + nu_ramp * (Nu - 1.0)
        for i in range(idx_c, nz):
            k_impl[i] = k_impl[i] * Nu_eff

    return idx_c, z_c, D_cond, D_conv, Tc, Ra, Nu, is_conv


# =============================================================================
# 5. TRIDIAGONAL SOLVER
# =============================================================================

@njit
def _thomas_solve(a, b, c, d, x, n):
    """
    Thomas algorithm for tridiagonal system.
    a: lower diagonal (length n, a[n-1] unused sentinel)
       a[k] couples row k+1 to row k. For row i, sub-diag entry is a[i-1].
    b: main diagonal (length n) — MODIFIED in-place during elimination
    c: upper diagonal (length n, c[n-1] unused sentinel)
    d: right-hand side (length n) — MODIFIED in-place during elimination
    x: solution output (length n)
    """
    # Forward elimination
    for i in range(1, n):
        if b[i - 1] == 0.0:
            continue
        m = a[i - 1] / b[i - 1]
        b[i] = b[i] - m * c[i - 1]
        d[i] = d[i] - m * d[i - 1]

    # Back substitution
    x[n - 1] = d[n - 1] / b[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = (d[i] - c[i] * x[i + 1]) / b[i]


# =============================================================================
# 6. STEFAN VELOCITY
# =============================================================================

@njit
def _stefan_velocity(T_col, nz, dz, q_ocean):
    """Stefan condition: db/dt = (k*dT/dz - q_ocean) / (rho*L).
    Uses bare 612/T conductivity (no porosity/salt), matching Solver.py:441."""
    T_base = T_col[nz - 1]
    k_basal = 612.0 / T_base
    # 2nd order one-sided gradient
    dTdz = (3.0 * T_col[nz - 1] - 4.0 * T_col[nz - 2] + T_col[nz - 3]) / (2.0 * dz)
    q_cond = k_basal * dTdz
    rho_base = _density_ice(T_base)
    return (q_cond - q_ocean) / (rho_base * 334000.0)
