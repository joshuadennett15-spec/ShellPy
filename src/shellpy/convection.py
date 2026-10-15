"""Stagnant-lid convection closure: the Deschamps & Vilella (2021, DV2021) scaling laws for a
mixed-heated ice shell, applied to the whole shell as in their application to Europa (their SI S3).

DV2021 predict, from the whole shell's Rayleigh number and viscosity contrast, the interior
temperature Tm (Eq. 21), the surface heat flux (Eq. 23) and the stagnant-lid thickness (Eq. 26).
A convecting shell takes heat from the ocean at the DV2021 bottom heat flux (Eq. 11), which
drives the Stefan condition. The column solver resolves the temperature, which sets the tidal
heating: the lid by conduction, and below it a conductivity multiplied by Nu so that the
convecting sublayer carries the DV2021 heat flux across the drop from the lid base (Eq. 19) to
the melting point.
"""
from dataclasses import dataclass

import numpy as np

from shellpy import constants as c
from shellpy import physics


@dataclass(frozen=True)
class ConvectionState:
    idx_c: int            # first grid node of the convecting sublayer
    D_cond: float         # stagnant-lid thickness (m); the whole shell when not convecting
    D_conv: float         # convecting sublayer thickness (m)
    T_lid: float          # lid-base temperature (K), DV2021 Eq. 19
    Tm: float             # interior temperature (K), DV2021 Eq. 21
    Ra: float             # effective Rayleigh number of the whole shell, at Tm (DV2021 Eq. 10)
    gamma: float          # viscosity-contrast parameter, Delta T / Delta T_v (DV2021 Eq. 22)
    Ur: float             # Urey ratio: heat generated in the shell / surface heat flux (DV2021 Eq. 12)
    q_bot: float          # convective heat flux into the ice at its base (W/m^2), DV2021 Eq. 11
    Nu: float             # factor on the sublayer conductivity
    is_convecting: bool   # convective surface flux exceeds the conductive one (DV2021 Eq. 14)
    interior_melting: bool = False   # convecting with Tm at the melting point: DV2021 predict partial melt


def interior_temperature(T_melt, dT, f, heat, p, Ra_of, k):
    """Tm from DV2021 Eq. 21 with constants k, by fixed-point iteration. heat is the shell's
    internal heating term H~ (1 + f + f^2) / 3 and Ra_of(Tm) the effective Rayleigh number.
    Capped at the melting point, where DV2021 predict partial melting."""
    Tm = T_melt
    for _ in range(100):
        heating = (k['c1'] + k['c2'] * f) * heat ** k['c4'] / Ra_of(Tm) ** k['c3']
        new = min(T_melt - c.DV_A1 / f**c.DV_A2 * physics.viscous_temperature_scale(Tm, p) + heating * dT, T_melt)
        if abs(new - Tm) < 1e-3:
            return new
        Tm = new
    raise RuntimeError(f"interior temperature did not converge (Tm = {Tm:.2f} K)")


def mean_conductivity(T_surf, T_melt, f_porosity):
    """Conductivity averaged over temperature between T_surf and T_melt (the conductivity integral
    over Delta T), with the porosity of ice colder than POROSITY_CURE_TEMP. DV2021 use one
    characteristic conductivity; this one makes their conductive limit match the conducting column."""
    cold = max(c.POROSITY_CURE_TEMP, T_surf)
    return 612.0 * (np.log(T_melt / T_surf) - f_porosity * np.log(cold / T_surf)) / (T_melt - T_surf)


def convection_state(z, H, T_surf, T_melt, q_tidal, p):
    """DV2021 stagnant-lid state of a shell of thickness H with tidal heating q_tidal (W/m^3) at depths z.

    Follows DV2021 SI S3: the scalings for Ur < 1 first; if they give a negative bottom heat flux,
    those for Ur > 1; if these give a positive one, the bottom heat flux is set to zero.
    """
    dT = T_melt - T_surf
    f = (c.RADIUS - H) / c.RADIUS
    k_ref = mean_conductivity(T_surf, T_melt, p['f_porosity'])
    q_carac = k_ref * dT / H
    H_tilde = np.average(q_tidal, weights=(c.RADIUS - z) ** 2) * H / q_carac   # Eq. 4, shell-mean heating
    heat = H_tilde * (1 + f + f**2) / 3                                         # heat made per unit surface area

    def Ra_of(T):
        rho = physics.density(T)
        kappa = physics.conductivity(T) / (rho * physics.specific_heat(T))
        return rho * c.GRAVITY * c.THERMAL_EXPANSION * dT * H**3 / (kappa * physics.diffusion_viscosity(T, p))

    def scalings(k):
        Tm = interior_temperature(T_melt, dT, f, heat, p, Ra_of, k)
        Ra = Ra_of(Tm)
        gamma = dT / physics.viscous_temperature_scale(Tm, p)
        phi_top = k['a'] * Ra**c.DV_B / gamma**c.DV_C                # Eq. 23
        d_lid = min(k['a_lid'] * gamma**c.DV_C / Ra**c.DV_B, 1.0)     # Eq. 26
        return Tm, Ra, gamma, phi_top, d_lid

    Tm, Ra, gamma, phi_top, d_lid = scalings(c.DV_UR_BELOW_1)
    if phi_top < heat:                                                 # bottom heat flux < 0: Ur > 1
        Tm, Ra, gamma, phi_top, d_lid = scalings(c.DV_UR_ABOVE_1)
        phi_top = min(phi_top, heat)
    Ur = heat / phi_top
    if not phi_top > f + (f + 2) / 6 * H_tilde:                        # Eq. 14: convection beats conduction
        return ConvectionState(len(z) - 1, H, 0.0, T_melt, Tm, Ra, gamma, Ur, 0.0, 1.0, False)
    q_top, q_bot = phi_top * q_carac, (phi_top - heat) / f**2 * q_carac   # Eq. 11
    # Eq. 19 with no heating in the lid; shortcut: capped one viscous temperature scale below the
    # melting point, since for Ur > 1 the Eq. 19 lid base can exceed it.
    T_lid = min(T_surf + dT * d_lid * phi_top, T_melt - physics.viscous_temperature_scale(Tm, p))
    D_cond = d_lid * H
    k_sub = physics.conductivity((T_lid + T_melt) / 2)
    # the sublayer carries q_bot at its base and q_top at the lid across T_melt - T_lid
    Nu = max((q_top + q_bot) / 2 * (H - D_cond) / (k_sub * (T_melt - T_lid)), 1.0)
    return ConvectionState(int(np.searchsorted(z, D_cond)), D_cond, H - D_cond, T_lid, Tm, Ra, gamma, Ur,
                           q_bot, Nu, True, Tm >= T_melt)


def conductivity_profile(T, z, H, T_melt, q_tidal, p):
    """Effective conductivity at each node, with porosity and convective enhancement.

    p holds the sampled shell parameters (rheology and f_porosity). Returns (k, ConvectionState).
    """
    k = physics.conductivity(T)
    k = np.where(T < c.POROSITY_CURE_TEMP, k * (1 - p['f_porosity']), k)
    state = convection_state(z, H, float(T[0]), T_melt, q_tidal, p)
    if state.is_convecting:
        k[state.idx_c:] *= state.Nu
    return k, state


def face_conductivity(k, state, z):
    """Conductivity of each face between neighbouring nodes at depths z: the harmonic mean
    (each node's half cell in series), except where the lid base lies inside the cell. There
    the lid share (plain k) and the sublayer share (Nu-enhanced k) follow the lid base's
    depth, so the convecting layer starts at D_cond rather than at the next node."""
    kf = 2.0 * k[:-1] * k[1:] / (k[:-1] + k[1:])
    i = state.idx_c
    if state.is_convecting and i > 0:
        f = (state.D_cond - z[i - 1]) / (z[i] - z[i - 1])   # lid fraction of the cell
        kf[i - 1] = 1.0 / (f / k[i - 1] + (1.0 - f) / k[i])
    return kf
