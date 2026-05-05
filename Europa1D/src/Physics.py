"""
Physics Engine for Europa Ice Shell Thermal Modeling

Implements material property models following Howell (2021) methodology.
All methods accept optional parameter overrides for Monte Carlo sampling.

Architecture:
    Stateless class with static methods for vectorized NumPy operations.
    Optional parameters default to None, which triggers use of Rheology/HeatFlux constants.
"""

import numpy as np
import numpy.typing as npt
from typing import Optional, Union
from math import gamma

# Import classes from constants.py
from constants import (Thermal, Rheology, HeatFlux, Planetary, Convection,
                       Porosity, FloatOrArray, ModelType)  # Convection used by convective_stress


class IcePhysics:
    """
    Physics engine for Ice Shell Simulations.
    Implements material property models from Howell (2021).

    All rheology methods accept optional parameter overrides to support
    Monte Carlo uncertainty propagation.
    """

    # =========================================================================
    # 1. RHEOLOGY & VISCOSITY
    # =========================================================================

    @staticmethod
    def gbs_strain_rate(
            T: FloatOrArray,
            d: float,
            sigma: float,
    ) -> FloatOrArray:
        """
        GBS creep strain rate (Goldsby & Kohlstedt 2001, Table 5, Eq. 15).

        Two temperature regimes split at T*=255 K.  All values in SI units
        (Pa, m, s, K).

        Args:
            T: Temperature (K)
            d: Grain size (m)
            sigma: Deviatoric stress (Pa)

        Returns:
            Strain rate (s^-1)
        """
        R = 8.314
        n, p = 1.8, 1.4
        T_arr = np.asarray(T, dtype=np.float64)

        A_lo, Q_lo = 6.18e-14, 49.0e3
        A_hi, Q_hi = 4.76e15, 192.0e3

        eps_lo = A_lo * d ** (-p) * sigma ** n * np.exp(-Q_lo / (R * T_arr))
        eps_hi = A_hi * d ** (-p) * sigma ** n * np.exp(-Q_hi / (R * T_arr))

        return np.where(T_arr < 255.0, eps_lo, eps_hi)

    @staticmethod
    def dislocation_strain_rate(
            T: FloatOrArray,
            sigma: float,
    ) -> FloatOrArray:
        """
        Dislocation creep strain rate (Goldsby & Kohlstedt 2001, Table 5, Eq. 13).

        Grain-size independent (p=0).  Two temperature regimes split at
        T*=258 K (note: different threshold from GBS).

        Args:
            T: Temperature (K)
            sigma: Deviatoric stress (Pa)

        Returns:
            Strain rate (s^-1)
        """
        R = 8.314
        n = 4.0
        T_arr = np.asarray(T, dtype=np.float64)

        A_lo, Q_lo = 4.0e-19, 60.0e3
        A_hi, Q_hi = 6.0e4, 181.0e3

        eps_lo = A_lo * sigma ** n * np.exp(-Q_lo / (R * T_arr))
        eps_hi = A_hi * sigma ** n * np.exp(-Q_hi / (R * T_arr))

        return np.where(T_arr < 258.0, eps_lo, eps_hi)

    @staticmethod
    def convective_stress(
            T_i: float,
            D_conv: float,
            Ra_i: float,
            Ra_crit: float = 1000.0,
    ) -> float:
        """
        Boundary-layer convective stress scale.

        sigma_conv = rho * g * alpha * DT_rh * delta_rh

        where:
            DT_rh  = 2.24 * R * T_i^2 / E_a   (rheological temperature drop)
            delta_rh = D_conv * (Ra_crit / Ra_i)^(1/3)  (boundary layer thickness)

        Uses constants: Planetary.GRAVITY (1.315 m/s^2),
        Convection.ALPHA_EXPANSION (1.6e-4 K^-1), Rheology.ACTIVATION_ENERGY_V
        (59400 J/mol), rho=920 kg/m^3.

        Args:
            T_i: Interior temperature (K)
            D_conv: Convective layer thickness (m)
            Ra_i: Rayleigh number of the convecting layer
            Ra_crit: Critical Rayleigh number (default 1000)

        Returns:
            Convective stress (Pa)
        """
        R = 8.314
        rho = 920.0
        g = Planetary.GRAVITY
        alpha = Convection.ALPHA_EXPANSION
        E_a = Rheology.ACTIVATION_ENERGY_V  # = Rheology.Q_V = 59400 J/mol

        DT_rh = 2.24 * R * T_i ** 2 / E_a
        delta_rh = D_conv * (Ra_crit / Ra_i) ** (1.0 / 3.0)

        return rho * g * alpha * DT_rh * delta_rh

    @staticmethod
    def composite_viscosity(
            T: FloatOrArray,
            d_grain: Optional[float] = None,
            d_del: Optional[float] = None,
            D0v: Optional[float] = None,
            D0b: Optional[float] = None,
            Q_diff: Optional[float] = None,
            Q_gbs: Optional[float] = None,
            d_molar: Optional[float] = None,
            p_grain: Optional[float] = None,
            creep_model: Optional[str] = None,
            sigma: Optional[float] = None,
    ) -> FloatOrArray:
        """
        Calculates grain-size-dependent viscosity.

        Two modes selected by *creep_model*:

        * ``"diffusion"`` (default / legacy) -- Nabarro-Herring + Coble
          diffusion creep (Howell 2021).  ``sigma`` is ignored.
        * ``"composite_gbs"`` -- Parallel sum of diffusion + GBS + dislocation
          creep (Goldsby & Kohlstedt 2001).  Requires ``sigma > 0``.

        Default p_grain=2.0 recovers Nabarro-Herring diffusion creep
        (Howell 2021).  Setting p_grain=1.4 follows the Goldsby &
        Kohlstedt (2001) GBS regime; p_grain=1.1 follows the Bayesian
        GSS fit of Prior et al. (2025, Nature Geoscience).

        Args:
            T: Temperature (K)
            d_grain: Ice grain size (m). If None, uses Rheology.GRAIN_SIZE default.
            d_del: Grain boundary width (m). If None, uses Rheology.GRAIN_WIDTH.
            D0v: Volume diffusion prefactor (m²/s). If None, uses Rheology.D0V_MEAN.
            D0b: Boundary diffusion prefactor (m²/s). If None, uses Rheology.D0B_MEAN.
            Q_diff: Volume diffusion activation energy (J/mol). If None, uses default.
            Q_gbs: Boundary diffusion activation energy (J/mol). If None, uses default.
            d_molar: Molar volume (m³/mol). If None, uses Rheology.MOLAR_VOLUME.
            p_grain: Grain-size exponent. If None, uses Rheology.GRAIN_EXPONENT
                     (default 2.0).
            creep_model: ``"diffusion"`` or ``"composite_gbs"``.  If None,
                         defaults to ``"diffusion"`` (backward compatible).
            sigma: Deviatoric stress (Pa).  Required when
                   ``creep_model="composite_gbs"``.

        Returns:
            Viscosity (Pa·s)
        """
        T_arr = np.asarray(T)
        T_safe = np.maximum(T_arr, 50.0)  # Prevent division by zero
        R = Rheology.GAS_CONSTANT

        d = d_grain if d_grain is not None else Rheology.GRAIN_SIZE
        delta = d_del if d_del is not None else Rheology.GRAIN_WIDTH
        D0v_use = D0v if D0v is not None else Rheology.D0V_MEAN
        D0b_use = D0b if D0b is not None else Rheology.D0B_MEAN
        Q_v = Q_diff if Q_diff is not None else Rheology.ACTIVATION_ENERGY_V
        Q_b = Q_gbs if Q_gbs is not None else Rheology.ACTIVATION_ENERGY_B
        d_vm = d_molar if d_molar is not None else Rheology.MOLAR_VOLUME
        p = p_grain if p_grain is not None else getattr(Rheology, 'GRAIN_EXPONENT', 2.0)

        # Diffusion coefficients (Howell 2021)
        Dv = D0v_use * np.exp(-Q_v / (R * T_safe))
        Db = D0b_use * np.exp(-Q_b / (R * T_safe))

        # Howell (2021) viscosity formulation
        prefactor = (42 * d_vm) / (R * T_safe * d ** 2)
        diff_term = Dv + (np.pi * delta / d) * Db
        eta_diffusion = 0.5 * (prefactor * diff_term) ** -1

        # Resolve creep model
        model = creep_model if creep_model is not None else "diffusion"

        if model == "diffusion":
            return np.clip(eta_diffusion, 1e12, 1e25)

        if model == "composite_gbs":
            if sigma is None or sigma <= 0:
                raise ValueError(
                    "composite_gbs creep model requires sigma > 0"
                )
            # Diffusion strain rate from existing eta_diffusion
            # Newtonian creep: sigma = 2*eta*eps => eps = sigma/(2*eta)
            eps_diff = sigma / (2.0 * eta_diffusion)

            # GBS and dislocation strain rates (Goldsby & Kohlstedt 2001)
            eps_gbs = IcePhysics.gbs_strain_rate(T_safe, d, sigma)
            eps_disl = IcePhysics.dislocation_strain_rate(T_safe, sigma)

            # Parallel sum: total strain rate = sum of all mechanisms
            eps_total = eps_diff + eps_gbs + eps_disl

            # Effective viscosity from total strain rate
            eta_composite = sigma / (2.0 * eps_total)
            return np.clip(eta_composite, 1e12, 1e25)

        raise ValueError(f"Unknown creep_model: {model!r}")

    @staticmethod
    def viscosity_simple(
            T: FloatOrArray,
            eta_ref: float = Rheology.VISCOSITY_REF,
            T_melt: float = Thermal.MELT_TEMP
    ) -> FloatOrArray:
        """
        Simple Arrhenius viscosity model (Frank-Kamenetskii approximation).

        Args:
            T: Temperature (K)
            eta_ref: Reference viscosity at melting point (Pa·s)
            T_melt: Melting temperature (K)

        Returns:
            Viscosity (Pa·s)
        """
        return Rheology.viscosity_F_K(T, eta_ref, T_melt)

    # =========================================================================
    # 2. THERMAL PROPERTIES
    # =========================================================================

    @staticmethod
    def effective_conductivity(
            T: FloatOrArray,
            porosity: float = 0.0,
            salt_fraction: float = 0.0,
            salt_scaling_factor: float = 1.0,
            porosity_cure_temp: Optional[float] = None,
            model: ModelType = None
    ) -> FloatOrArray:
        """
        Calculates effective thermal conductivity with porosity and salt corrections.

        Porosity:  k_eff = k_ice * (1 - f)  [for T < T_curing]
        Salt:      k_eff = (1-f_s)k_ice + f_s(k_ice*B_k)

        Args:
            T: Temperature (K)
            porosity: Porosity fraction (0-1)
            salt_fraction: Salt fraction (0-1)
            salt_scaling_factor: Salt conductivity scaling factor B_k
            model: Conductivity model ("Carnahan", "Howell", etc.)

        Returns:
            Effective thermal conductivity (W/m·K)
        """
        T_arr = np.asarray(T)
        k_base = Thermal.conductivity(T_arr, model=model)

        # Porosity correction (cold, brittle, conductive upper shell)
        if porosity > 0:
            T_phi = porosity_cure_temp if porosity_cure_temp is not None else Thermal.POR_CUR_TEMP_MEAN
            is_porous = T_arr < T_phi
            correction = (1 - porosity)
            k_base = np.where(is_porous, k_base * correction, k_base)

        # Salt scaling
        if salt_fraction > 0:
            k_base = k_base * (1 + salt_fraction * (salt_scaling_factor - 1.0))

        return k_base

    @staticmethod
    def thermal_diffusivity(
            T: FloatOrArray,
            model: ModelType = "Carnahan"
    ) -> FloatOrArray:
        """
        Calculates thermal diffusivity: α = k / (ρ · cp).

        Returns:
            Thermal diffusivity (m²/s)
        """
        k = Thermal.conductivity(T, model=model)
        rho = Thermal.density_ice(T)
        cp = Thermal.specific_heat(T)

        return k / (rho * cp)

    @staticmethod
    def thermal_inertia(
            T: FloatOrArray,
            model: ModelType = "Carnahan"
    ) -> FloatOrArray:
        """
        Calculates thermal inertia: Γ = √(k · ρ · cp).
        Controls the rate of temperature change in response to surface forcing.

        Returns:
            Thermal inertia (J/(m²·K·s^0.5)), also known as "tiu"
        """
        k = Thermal.conductivity(T, model)
        rho = Thermal.density_ice(T)
        cp = Thermal.specific_heat(T)

        return np.sqrt(k * rho * cp)

    # =========================================================================
    # 3. TIDAL MECHANICS
    # =========================================================================

    @staticmethod
    def tidal_heating(
            T: FloatOrArray,
            epsilon_0: Optional[float] = None,
            mu_ice: Optional[float] = None,
            use_composite_viscosity: bool = False,
            eta_ref: float = Rheology.VISCOSITY_REF,
            d_grain: Optional[float] = None,
            Q_diff: Optional[float] = None,
            Q_gbs: Optional[float] = None,
            D0v: Optional[float] = None,
            D0b: Optional[float] = None,
            d_del: Optional[float] = None,
            p_grain: Optional[float] = None,
    ) -> FloatOrArray:
        """
        Calculates volumetric tidal dissipation q_dot (W/m³).

        Supports both Maxwell and Andrade (transient creep) rheology.
        Andrade formulation based on McCarthy et al. (2011) and Renaud & Henning (2018).

        Maxwell:
            q_dot = (ε₀² · ω² · η) / [2 · (1 + (ω² · η² / μ²))]
            
        Andrade:
            Calculates complex compliance J*(w), where dissipation is proportional
            to the imaginary component of the shear modulus Im(1/J*).
            
        Args:
            T: Temperature (K)
            epsilon_0: Tidal strain amplitude. If None, uses HeatFlux.TIDAL_STRAIN.
            mu_ice: Ice shear modulus (Pa). If None, uses Rheology.RIGIDITY_ICE.
            use_composite_viscosity: Use composite flow law (default: False).
            eta_ref: Reference viscosity for simple model (Pa·s).
            d_grain: Grain size (m). Required if use_composite_viscosity=True.
            Q_diff: Diffusion creep activation energy (J/mol).
            Q_gbs: Boundary diffusion activation energy (J/mol).
            D0v: Volume diffusion prefactor (m²/s).
            D0b: Boundary diffusion prefactor (m²/s).
            d_del: Grain boundary width (m).

        Returns:
            Volumetric heating rate (W/m³)
        """
        # Use overrides or defaults
        strain = epsilon_0 if epsilon_0 is not None else HeatFlux.TIDAL_STRAIN
        mu = mu_ice if mu_ice is not None else Rheology.RIGIDITY_ICE

        # Select viscosity model
        if use_composite_viscosity:
            eta = IcePhysics.composite_viscosity(
                T,
                d_grain=d_grain,
                d_del=d_del,
                D0v=D0v,
                D0b=D0b,
                Q_diff=Q_diff,
                Q_gbs=Q_gbs,
                p_grain=p_grain,
            )
        else:
            eta = IcePhysics.viscosity_simple(T, eta_ref)

        omega = Planetary.ORBITAL_FREQ
        
        # Check config for Rheology Model
        model_type = Rheology.MODEL

        if model_type == "Andrade":
            # Andrade transient creep parameters
            alpha = Rheology.ANDRADE_ALPHA
            zeta = Rheology.ANDRADE_ZETA
            
            # Compliance J (1/mu)
            J_elastic = 1.0 / mu
            
            # Andrade term incorporates forcing frequency and relaxation time (eta/mu)
            tau = eta / mu
            andrade_term = omega * tau * zeta
            
            # Avoid divide-by-zero
            andrade_term = np.clip(andrade_term, 1e-100, None)
            
            # Complex compliance J*(w) = J_real - i*J_imag
            # Based on McCarthy et al. 2011 pseudo-periodic compliance
            const_term = J_elastic * (andrade_term ** -alpha) * gamma(1 + alpha)
            
            J_real = J_elastic + const_term * np.cos(alpha * np.pi / 2.0)
            J_imag = J_elastic * (omega * tau) ** -1 + const_term * np.sin(alpha * np.pi / 2.0)
            
            # Dissipative shear modulus Im(G*) = J_imag / |J*|^2
            G_imag = J_imag / (J_real**2 + J_imag**2)
            
            # Volumetric Heating: q = 0.5 * omega * strain^2 * Im(G*)
            return 0.5 * omega * (strain ** 2) * G_imag
            
        else: # Default: Maxwell
            # Maxwell viscoelastic dissipation (time-averaged over tidal cycle)
            # q_vol = ε₀² ω² η / [2·(1 + (ωη/μ)²)]
            # Factor of 1/2 from time-averaging sin²(ωt) over one period
            numerator = (strain ** 2) * (omega ** 2) * eta
            denominator = 2.0 * (1.0 + ((omega ** 2) * (eta ** 2) / (mu ** 2)))
            return numerator / denominator

    # =========================================================================
    # 4. BOUNDARY CONDITIONS
    # =========================================================================

    @staticmethod
    def basal_melting_point(
            ice_thickness_m: FloatOrArray,
            rho_ice: float = 917.0,
            g: float = Planetary.GRAVITY
    ) -> FloatOrArray:
        """
        Returns pressure-dependent melting temperature at the ice base.

        Args:
            ice_thickness_m: Ice shell thickness (m)
            rho_ice: Ice density (kg/m³)
            g: Surface gravity (m/s²)

        Returns:
            Melting temperature (K)
        """
        thickness = np.asarray(ice_thickness_m)
        pressure_pa = rho_ice * g * thickness
        return Thermal.MELT_TEMP + (Thermal.CLAUSIUS_CLAPEYRON * pressure_pa)

    @staticmethod
    def basal_pressure(
            ice_thickness_m: FloatOrArray,
            rho_ice: float = 917.0,
            g: float = Planetary.GRAVITY
    ) -> FloatOrArray:
        """
        Calculates hydrostatic pressure at the ice-ocean interface.

        Returns:
            Pressure (Pa)
        """
        thickness = np.asarray(ice_thickness_m)
        return rho_ice * g * thickness

    @staticmethod
    def stefan_velocity(
            T_profile: npt.NDArray[np.float64],
            dz: float,
            q_ocean: float,
            k_basal: Optional[float] = None
    ) -> float:
        """
        Calculates the freezing front velocity using the Stefan condition.

        Args:
            T_profile: Temperature profile from surface to base (K)
            dz: Grid spacing (m)
            q_ocean: Ocean heat flux (W/m²)
            k_basal: Thermal conductivity at base (W/m·K). If None, computed from T.

        Returns:
            Freezing velocity (m/s). Positive = freezing, negative = melting.
        """
        T_base = T_profile[-1]

        if k_basal is None:
            k_basal = Thermal.conductivity(T_base)

        # Temperature gradient at base (2nd order one-sided difference)
        if len(T_profile) >= 3:
            dT_dz = (3 * T_profile[-1] - 4 * T_profile[-2] + T_profile[-3]) / (2 * dz)
        else:
            dT_dz = (T_profile[-1] - T_profile[-2]) / dz

        # Heat flux conducted into ice from base
        q_conducted = k_basal * dT_dz

        # Stefan condition: v = (q_conducted - q_ocean) / (ρ · L)
        rho_ice = Thermal.density_ice(T_base)
        L = Thermal.LATENT_HEAT

        return (q_conducted - q_ocean) / (rho_ice * L)
