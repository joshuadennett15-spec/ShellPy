"""Physical constants for Europa's ice shell, in SI units.

Values follow Howell (2021) unless a different source is noted.
"""
import numpy as np

# Europa
RADIUS = 1_561_000.0                 # m
GRAVITY = 1.315                      # m/s^2
ORBITAL_FREQ = 2.047e-5              # rad/s, tidal forcing frequency
AREA = 4 * np.pi * RADIUS**2         # m^2, surface area

# Ice
MELT_TEMP = 273.0                    # K, melting point at zero pressure
CLAUSIUS_CLAPEYRON = -7.4e-8         # K/Pa, slope of the melting curve
LATENT_HEAT = 334_000.0              # J/kg
ICE_DENSITY_REF = 917.0              # kg/m^3, at the melting point
THERMAL_EXPANSION = 1.6e-4           # 1/K
POROSITY_CURE_TEMP = 150.0           # K, ice warmer than this has lost its porosity

# Rheology
GAS_CONSTANT = 8.314                 # J/(mol K)
MOLAR_VOLUME = 1.97e-5               # m^3/mol
ANDRADE_ALPHA = 0.2                  # Andrade transient-creep exponent
ANDRADE_ZETA = 1.0                   # Andrade-to-Maxwell timescale ratio

# Stagnant-lid convection (Deschamps & Vilella 2021, DV2021)
RA_CRIT = 1000.0                     # critical Rayleigh number for onset
C1_DESCHAMPS = 1.43                  # DV2021 Eq. 18
C2_DESCHAMPS = -0.03                 # DV2021 Eq. 18
THETA_LID = 2.24                     # lid-base temperature drop in units of the viscous temperature scale
