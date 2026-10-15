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

# Stagnant-lid convection: Deschamps & Vilella (2021, DV2021) Table 2.
# Calibrated on 1e4 <= viscosity ratio <= 1e8 (gamma 9.2-18.4) and 2e5 <= Ra_eff <= 2e8.
DV_A1, DV_A2 = 1.23, 1.5                          # interior temperature, Eq. 21
DV_B, DV_C = 0.27, 1.21                           # Ra and gamma exponents, Eqs. 23 and 26
# Eq. 21 internal-heating term (c1-c4), Eq. 23 surface flux (a), Eq. 26 lid thickness (a_lid),
# for bottom heat flux >= 0 (Urey ratio Ur < 1) and < 0 (Ur > 1)
DV_UR_BELOW_1 = {'c1': 3.5, 'c2': -2.3, 'c3': 0.25, 'c4': 1.0, 'a': 1.46, 'a_lid': 0.633}
DV_UR_ABOVE_1 = {'c1': 4.4, 'c2': -3.0, 'c3': 1 / 3, 'c4': 1.72, 'a': 1.57, 'a_lid': 0.667}
