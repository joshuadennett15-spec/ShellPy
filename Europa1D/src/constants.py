
import numpy as np
import numpy.typing as npt
from dataclasses import dataclass
from typing import Final, Literal, Union

FloatOrArray = Union[float, npt.NDArray[np.float64]]
ModelType = Literal["Carnahan", "Howell", "CBE", "Andrade", "Maxwell"]

from ConfigManager import ConfigManager
cfg = ConfigManager()

# Physical constants
@dataclass (frozen=True)
class Planetary:
    """Section 1: Standard Planetary Values"""
    RADIUS: Final[float] = cfg.get("planetary", "RADIUS", 1_561_000.00)
    MASS: Final[float] = cfg.get("planetary", "MASS", 4.80e22)
    GRAVITY: Final[float] = cfg.get("planetary", "GRAVITY", 1.315)
    ORBITAL_FREQ: Final[float] = cfg.get("planetary", "ORBITAL_FREQ", 2.047e-5)
    ECCENTRICITY: Final[float] = cfg.get("planetary", "ECCENTRICITY", 0.0101)
    ROTATION_PERIOD: Final[float] = 3.55 * 24 * 3600    # seconds (synchronous with orbit)
    AREA: Final[float] = 4 * np.pi * (RADIUS ** 2)

@dataclass(frozen=True)
class Thermal:
    """Section 2: Thermal & Material Parameters"""
    BOLTZMANN: Final[float] = 5.67e-8                    # W/m²·K⁴ (σ) Stefan-Boltzmann constant
    LATENT_HEAT: Final[float] = 334_000.0                # J/kg (L)
    EMISSIVITY: Final[float] = 0.94                      # Surface emissivity (ε)
    MELT_TEMP: Final[float] = cfg.get("thermal", "MELT_TEMP", 273.0)
    POR_CUR_TEMP_MEAN: Final[float] = cfg.get("thermal", "POR_CUR_TEMP_MEAN", 150.0)
    SPECIFIC_HEAT: Final[float] = 2000.0                 # J/kg K (Howell 2021)
    THERMAL_EXPANSION: Final[float] = 1.6e-4             # K^-1 (Howell 2021)
    ICE_DENSITY_REF: Final[float] = 917.0                # kg/m^3 at melt (Howell 2021)

    # Dirichlet Boundary Conditions (Monte Carlo Distribution Range)
    SURFACE_TEMP_MEAN: Final[float] = cfg.get("thermal", "SURFACE_TEMP_MEAN", 104.0)
    SURFACE_TEMP_STD: Final[float] = 7.0                 # K

    # Clausius-Clapeyron for pressure dependent melting
    CLAUSIUS_CLAPEYRON: Final[float] = -7.4e-8           # dT_m/dP (K/Pa)

    @staticmethod
    def conductivity(T: FloatOrArray, model: ModelType = None) -> FloatOrArray:
        """
        Returns Thermal Conductivity k(T) in W/m K.

        When *model* is None the active model is read from config.json
        (thermal.CONDUCTIVITY_MODEL, default "Carnahan").
        """
        if model is None:
            from ConfigManager import ConfigManager
            model = ConfigManager.get("thermal", "CONDUCTIVITY_MODEL", "Carnahan")

        T_arr = np.asarray(T)                   # Treat T as an array for safe division

        if model == "Carnahan":
            return 612.0 / T_arr
        elif model == "Howell":
            return 567.0 / T_arr
        elif model == "CBE":
            return np.full_like(T_arr, 3.2) if T_arr.ndim > 0 else 3.2

        raise ValueError(f"Unknown model: {model}")

    @staticmethod
    def specific_heat(T: FloatOrArray) -> FloatOrArray:
        """
        Specific Heat Capacity (cp) as a function of temperature.
        Linear fit to ice Ih data: cp ≈ 7.49·T + 90 J/kg·K
        (gives ~839 J/kg·K at 100 K, ~2135 J/kg·K at 273 K)
        Reference: Giauque & Stout (1936), consistent with Feistel & Wagner (2006).
        """
        T_arr = np.asarray(T)
        cp = 7.49 * T_arr + 90.0
        return cp if T_arr.ndim > 0 else float(cp)

    @staticmethod
    def density_ice(T: FloatOrArray, use_cbe_constant: bool = False) -> FloatOrArray:
        """
        Returns Ice Density rho(T) in kg/m^3.
        Where use_cbe_constant is True, returns 917.0.
        otherwise uses Howell (2021): rho = rho0 * (1 + alpha * (T_m - T))
        """
        T_arr = np.asarray(T)
        if use_cbe_constant:

            return np.full_like(T_arr, 917.0) if T_arr.ndim > 0 else 917.0

        return Thermal.ICE_DENSITY_REF * (1 + Thermal.THERMAL_EXPANSION * (Thermal.MELT_TEMP - T_arr))

@dataclass(frozen=True)
class Rheology:
    """Section 3: Rheology & Viscosity"""
    GRAIN_SIZE: Final[float] = cfg.get("rheology", "GRAIN_SIZE", 1.0e-3)
    GRAIN_WIDTH: Final[float] = cfg.get("rheology", "GRAIN_WIDTH", 7.13e-10)
    TRANSITION_TEMP: Final[float] = 230.0       # K (T_grain_switch)
    D_GRAIN_COLD: Final[float] = 1.0e-3         # 1 mm (cold lid)
    D_GRAIN_WARM: Final[float] = 1.0e-4         # 0.1 mm (warm ductile)

    D0V_MEAN: Final[float] = 9.1e-4            # m^2/s (Howell 2021)
    D0B_MEAN: Final[float] = 8.4e-4            # m^2/s (Howell 2021)
    D0_VAR: Final[float] = 0.033               # Fractional variance (Howell 2021)
    ACTIVATION_ENERGY_V: Final[float] = 59.4e3  # J/mol (Q_v)
    ACTIVATION_ENERGY_B: Final[float] = 49.0e3  # J/mol (Q_b)
    GAS_CONSTANT: Final[float] = 8.314          # J/mol K (R)
    MOLAR_VOLUME: Final[float] = 1.97e-5        # m^3/mol (Howell 2021)

    # Reference viscosity - Green et al. (2021) uses 5e13, Howell uses 10^14.7
    VISCOSITY_REF: Final[float] = cfg.get("rheology", "VISCOSITY_REF", 5.0e13)
    VISCOSITY_REF_HOWELL: Final[float] = 10 ** 14.7  # Pa·s (Howell 2021 for comparison)
    RIGIDITY_ICE: Final[float] = 3.3e9          # Pa (mu) - Green uses 3.3e9

    # Stress Exponent
    STRESS_EXPONENT: Final[float] = 1.8  # (n) for GSS creep
    
    # Andrade Specific
    MODEL: Final[str] = cfg.get("rheology", "model", "Maxwell")
    ANDRADE_ALPHA: Final[float] = cfg.get("rheology", "andrade_alpha", 0.2)
    ANDRADE_ZETA: Final[float] = cfg.get("rheology", "andrade_zeta", 1.0)

    @staticmethod
    def viscosity_F_K(T: FloatOrArray, eta_m: float, T_melt: float) -> FloatOrArray:
        """
        Returns viscosity using Frank-Kamenetskii approximation
        eta(T) = eta_m * exp((Q_v/R) * (1/T - 1/T_melt))
        """
        Q = Rheology.ACTIVATION_ENERGY_V
        R = Rheology.GAS_CONSTANT
        T_arr = np.asarray(T)

        return eta_m * np.exp((Q / R) * (1/T_arr - 1/T_melt))

@dataclass(frozen=True)
class HeatFlux:
    """Section 4: Heat Flux & dissipation"""
    # Fluxes in W/m² (Converted from mW/m²)
    SURFACE_FLUX_CBE: Final[float] = 24.5e-3            # W/m²
    RADIOGENIC_FLUX: Final[float] = cfg.get("heat_flux", "RADIOGENIC_FLUX", 7.4e-3)
    TIDAL_SILICATE_FLUX: Final[float] = cfg.get("heat_flux", "TIDAL_SILICATE_FLUX", 3.5e-3)
    TIDAL_ICE_FLUX_CBE: Final[float] = 30.9e-3          # W/m²

    TIDAL_STRAIN: Final[float] = cfg.get("heat_flux", "TIDAL_STRAIN", 1.0e-5)
    TIDAL_STRAIN_MIN: Final[float] = 2.0e-6             # W/m²
    TIDAL_STRAIN_MAX: Final[float] = 3.4e-5             # W/m²

    @property
    def total_ocean_flux(self) -> float:
        """
        Returns sum of radiogenic and silicate tidal flux (W/m²)
        """

        return self.RADIOGENIC_FLUX + self.TIDAL_SILICATE_FLUX

@dataclass(frozen=True)
class Convection:
    """
    Section 5: Ductile lid - Convection parameters
    
    Green et al. (2021) / Solomatov & Moresi (2000) scaling laws.
    """
    # Critical Rayleigh number - lowered to match Green et al. behavior
    # Green et al. doesn't use Ra_crit threshold; convection always active if layer exists
    RA_CRIT: Final[float] = cfg.get("convection", "RA_CRIT", 1000.0)

    # Which Nu(Ra) scaling law to use.  Each law was calibrated under a specific
    # viscosity model, so this selector also controls Ra computation:
    #   "green"                 – Green et al. (2021) C·Ra^(1/3) with FK viscosity
    #   "howell"                – Pre-Phase-2 Howell (2021) simple parameterization
    #   "isoviscous_benchmark"  – 0.088·Ra^0.28, diagnostic only (violates Δη~1e7)
    #   "dv2021"                – Deschamps & Vilella (2021) [not yet implemented]
    _VALID_NU_SCALING: Final[tuple] = ("green", "howell", "isoviscous_benchmark", "dv2021")
    NU_SCALING: Final[str] = cfg.get("convection", "NU_SCALING", "green")
    
    # Nusselt number scaling: Nu = C * Ra^beta * ((Ti-Tc)/DT)^zeta
    # Solomatov & Moresi (2000) / Green et al. (2021) values:
    NU_PREFACTOR: Final[float] = cfg.get("convection", "NU_PREFACTOR", 0.3446)
    BETA_NU: Final[float] = 0.333                       # xi = 1/3 (Nu exponent)
    ZETA_NU: Final[float] = 1.333                       # zeta = 4/3 (internal heating correction)
    
    # Interior temperature scaling (Deschamps & Vilella 2021)
    GAMMA_RA: Final[float] = 0.25                       # gamma (Ra exponent for Ti)
    BETA_H: Final[float] = 0.75                         # beta (heating exponent for Ti)
    D_PREFACTOR: Final[float] = 1.0                     # D prefactor for Ti scaling
    
    # Rheological parameters for Tc calculation
    C1_DESCHAMPS: Final[float] = 1.43                   # Deschamps Eq.16 constant
    C2_DESCHAMPS: Final[float] = -0.03                  # Deschamps Eq.16 constant
    THETA_LID: Final[float] = 2.24                      # DTe/DTv ratio (was 2.21)

    # Thermal expansion coefficient (unified with Thermal.THERMAL_EXPANSION)
    ALPHA_EXPANSION: Final[float] = 1.6e-4              # K⁻¹ (Howell 2021)


# Import-time validation — fires on first `from constants import ...`
if Convection.NU_SCALING not in Convection._VALID_NU_SCALING:
    raise ValueError(
        f"NU_SCALING={Convection.NU_SCALING!r} not in {Convection._VALID_NU_SCALING}"
    )
# dv2021 is now implemented in Convection.py — no import guard needed.


@dataclass(frozen=True)
class Porosity:
    """Section 6: Porosity - Upper Shell Dynamics and Salinity"""
    # Porosity sampled in Monte Carlo simulation
    POROSITY_MEAN: Final[float] = 0.1                   # Mean Porosity fraction
    POROSITY_MAX: Final[float] = 0.3                    # Maximum Porosity

    # Salinity Scaling
    B_K_DEFAULT: Final[float] = 1.0                     # Salt Conductivity Factor
    F_SALT_DEFAULT: Final[float] = 0.0                  # Salt Fraction