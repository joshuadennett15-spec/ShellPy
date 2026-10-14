"""Material properties of ice: conductivity, density, heat capacity, melting point,
viscosity, tidal heating and the Stefan condition at the ice-ocean interface.

All functions take temperatures in K and work on floats or numpy arrays.
"""
from math import gamma

import numpy as np

from shellpy import constants as c


def conductivity(T):
    """Thermal conductivity of pure ice, k = 612/T W/m/K (Carnahan et al. 2021)."""
    return 612.0 / np.asarray(T)


def density(T):
    """Ice density with thermal expansion, kg/m^3 (Howell 2021)."""
    return c.ICE_DENSITY_REF * (1 + c.THERMAL_EXPANSION * (c.MELT_TEMP - np.asarray(T)))


def specific_heat(T):
    """Heat capacity of ice Ih, linear fit cp = 7.49 T + 90 J/kg/K."""
    return 7.49 * np.asarray(T) + 90.0


def melting_point(H):
    """Melting temperature (K) at the base of a shell of thickness H (m)."""
    return c.MELT_TEMP + c.CLAUSIUS_CLAPEYRON * (c.ICE_DENSITY_REF * c.GRAVITY * np.asarray(H))


def diffusion_viscosity(T, d_grain, d_del, D0v, D0b, Q_v, Q_b):
    """Diffusion-creep viscosity (Pa s): volume plus grain-boundary diffusion (Howell 2021)."""
    R = c.GAS_CONSTANT
    T = np.asarray(T)
    Dv = D0v * np.exp(-Q_v / (R * T))
    Db = D0b * np.exp(-Q_b / (R * T))
    rate = (42 * c.MOLAR_VOLUME) / (R * T * d_grain**2) * (Dv + (np.pi * d_del / d_grain) * Db)
    return np.clip(0.5 / rate, 1e12, 1e25)


def fk_viscosity(T, T_melt):
    """Frank-Kamenetskii viscosity (Pa s) anchored at VISCOSITY_REF at the melting point."""
    return c.VISCOSITY_REF * np.exp((c.Q_FK / c.GAS_CONSTANT) * (1 / np.asarray(T) - 1 / T_melt))


def andrade_heating(eta, epsilon_0, mu_ice):
    """Volumetric tidal heating (W/m^3) of an Andrade solid: 0.5 w eps0^2 Im(G*).

    eta is the steady-state viscosity, epsilon_0 the tidal strain amplitude and
    mu_ice the shear modulus (Pa). Complex compliance after McCarthy et al. (2011).
    """
    w, a = c.ORBITAL_FREQ, c.ANDRADE_ALPHA
    J = 1.0 / mu_ice
    tau = eta / mu_ice                                    # Maxwell time
    transient = J * (w * tau * c.ANDRADE_ZETA) ** -a * gamma(1 + a)
    J_real = J + transient * np.cos(a * np.pi / 2.0)
    J_imag = J * (w * tau) ** -1 + transient * np.sin(a * np.pi / 2.0)
    return 0.5 * w * epsilon_0**2 * (J_imag / (J_real**2 + J_imag**2))


def stefan_velocity(q_ice, q_ocean, T_base):
    """Freezing-front velocity (m/s, positive = freezing): (q_ice - q_ocean) / (rho L).

    q_ice is the heat flux conducted up into the ice at its base, q_ocean the flux
    the ocean delivers (both W/m^2).
    """
    return (q_ice - q_ocean) / (density(T_base) * c.LATENT_HEAT)
