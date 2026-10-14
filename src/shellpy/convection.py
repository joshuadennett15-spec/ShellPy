"""Stagnant-lid convection closure (Deschamps & Vilella 2021, applied as in Green et al. 2021).

The shell is split where the temperature profile reaches the lid-base temperature T_c:
a conductive lid above and a convecting sublayer below. The sublayer's conductivity is
multiplied by the Nusselt number, so a 1D conduction solver carries the convective flux.
"""
from dataclasses import dataclass

import numpy as np

from shellpy import constants as c
from shellpy import physics


@dataclass(frozen=True)
class ConvectionState:
    idx_c: int            # first grid node of the convecting sublayer
    D_cond: float         # conductive lid thickness (m)
    D_conv: float         # convecting sublayer thickness (m)
    T_c: float            # lid-base temperature (K)
    Ti: float             # interior temperature of the sublayer (K)
    Ra: float             # Rayleigh number of the sublayer
    Nu: float             # Nusselt number applied to the sublayer conductivity
    is_convecting: bool   # Ra >= RA_CRIT


def lid_temperatures(T_melt, T_surf, Q_v):
    """Lid-base temperature T_c and interior temperature Ti (K).

    Ti from DV2021 Eq. 18; T_c = Ti - THETA_LID * R Ti^2 / Q_v. Both are kept
    at least 1 K inside the surface-to-melting-point window.
    """
    R = c.GAS_CONSTANT
    B = Q_v / (2 * R * c.C1_DESCHAMPS)
    Ti = B * (np.sqrt(1.0 + (2.0 / B) * (T_melt - c.C2_DESCHAMPS * (T_melt - T_surf))) - 1.0)
    Ti = float(np.clip(Ti, T_surf + 1.0, T_melt - 1.0))
    T_c = Ti - c.THETA_LID * (R * Ti**2 / Q_v)
    return float(np.clip(T_c, T_surf + 1.0, T_melt - 1.0)), Ti


def rayleigh_nusselt(T_c, T_melt, D_conv, Q_v):
    """Rayleigh number of the sublayer and its Nusselt number from DV2021 Eq. 23.

    Ra uses the Frank-Kamenetskii viscosity at the sublayer's mean temperature.
    Nu is DV2021's surface heat flux Phi_top with no internal heating, for which
    their Eq. 21 gives the interior temperature 1 - 1.23/gamma in closed form.
    """
    DT = T_melt - T_c
    T_mean = (T_melt + T_c) / 2
    rho = physics.density(T_mean)
    kappa = physics.conductivity(T_mean) / (rho * physics.specific_heat(T_mean))
    buoyancy = rho * c.GRAVITY * c.THERMAL_EXPANSION * DT * D_conv**3
    Ra = buoyancy / (kappa * physics.fk_viscosity(T_mean, T_melt))
    Ra_surf = buoyancy / (kappa * physics.fk_viscosity(T_c, T_melt))
    gamma = max(Q_v / c.GAS_CONSTANT * (1.0 / T_c - 1.0 / T_melt), 1.0)   # ln of the viscosity contrast
    Ra_eff = Ra_surf * np.exp(gamma * (1.0 - 1.23 / gamma))
    Nu = max(1.46 * Ra_eff**0.27 / gamma**1.21 * c.FK_CORRECTION, 1.0)
    return float(Ra), float(Nu)


def convection_state(T, z, H, T_melt, Q_v):
    """Locate the lid base in the temperature profile T at depths z and characterise the sublayer."""
    T_c, Ti = lid_temperatures(T_melt, float(T[0]), Q_v)
    warm = np.flatnonzero(T >= T_c)
    if warm.size == 0:
        return ConvectionState(len(T) - 1, H, 0.0, T_c, Ti, 0.0, 1.0, False)
    i = int(warm[0])
    if i > 0 and T[i] > T[i - 1]:   # interpolate the depth where T == T_c
        D_cond = z[i - 1] + (T_c - T[i - 1]) / (T[i] - T[i - 1]) * (z[i] - z[i - 1])
    else:
        D_cond = z[i]
    D_conv = H - D_cond
    if D_conv <= 0:
        return ConvectionState(i, D_cond, 0.0, T_c, Ti, 0.0, 1.0, False)
    Ra, Nu = rayleigh_nusselt(T_c, T_melt, D_conv, Q_v)
    return ConvectionState(i, D_cond, D_conv, T_c, Ti, Ra, Nu, Ra >= c.RA_CRIT)


def conductivity_profile(T, z, H, T_melt, p):
    """Effective conductivity at each node, with porosity, salt and convective enhancement.

    p holds the sampled shell parameters (Q_v, f_porosity, T_phi, f_salt, B_k).
    Returns (k, ConvectionState).
    """
    k = physics.conductivity(T)
    k = np.where(T < p['T_phi'], k * (1 - p['f_porosity']), k)   # ice colder than T_phi is porous
    k = k * (1 + p['f_salt'] * (p['B_k'] - 1.0))
    state = convection_state(T, z, H, T_melt, p['Q_v'])
    if state.is_convecting:
        k[state.idx_c:] *= state.Nu
    return k, state


def face_conductivity(k):
    """Harmonic-mean conductivity between neighbouring nodes (flux-conservative)."""
    return 2.0 * k[:-1] * k[1:] / (k[:-1] + k[1:])
