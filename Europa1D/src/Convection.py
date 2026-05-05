"""
Stagnant - Lid convection parameterization for Europa's ice shell

Implements Nusselt number (Nu) scaling to adjust effective conductivity in the basal convective layer.
When the ice shell exceeds a critical depth such that the local rayleigh No. > Critical Rayleigh No.,
it transitions from purely conductive to stagnant lid convection.

Shell Structure:
    - Cold conductive lid (thickness z_c)
    - Warm, near isothermal convective sublayer below

Convection significantly increases heat transfer efficiency.

Phase 2 Parameterized Convection (Green et al. 2021, Deschamps & Vilella 2021):
    - Dynamic scanning of temperature profile to find conductive/convective interface
    - Harmonic mean averaging for flux-conservative differencing at interfaces
    - Explicit tracking of D_cond and D_conv at each timestep

References:
    - Barr & Showman (2009): Nu scaling laws
    - Howell (2021): Europa-specific parameters
    - Solomatov (1995): Stagnant-lid theory
    - Green et al. (2021): Parameterized convection methodology
    - Deschamps & Vilella (2021): Nu-Ra scaling laws
"""

import numpy as np
import numpy.typing as npt
from typing import Tuple, Optional, NamedTuple
from dataclasses import dataclass
from scipy.optimize import brentq

from constants import Thermal, Planetary, Rheology, Convection as ConvectionConstants, FloatOrArray
from Physics import IcePhysics
from ConfigManager import ConfigManager


# =============================================================================
# DATA STRUCTURES FOR CONVECTION STATE
# =============================================================================

@dataclass
class ConvectionState:
    """
    Container for convection diagnostics at a given timestep.
    
    Attributes:
        idx_c: Grid index of conductive/convective interface
        z_c: Depth of conductive lid (m)
        D_cond: Conductive lid thickness (m)
        D_conv: Convective layer thickness (m)
        T_c: Critical transition temperature (K)
        Ra: Rayleigh number of convective layer
        Nu: Nusselt number
        is_convecting: Whether convection is active
    """
    idx_c: int
    z_c: float
    D_cond: float
    D_conv: float
    T_c: float
    Ti: float
    Ra: float
    Nu: float
    is_convecting: bool
    nu_scaling: str = "green"

class IceConvection:
    """
    Stateless engine for stagnant lid convection parameterization.
    All methods are static and vectorized for NumPy array aperation
    """
    LID_FRACTION: float = 0.88

    @staticmethod
    def _fallback_howell_transition(T_melt: float) -> float:
        """Fallback conductive-base temperature when the analytic Howell root is invalid."""
        safe_melt = float(T_melt) if np.isfinite(T_melt) else Thermal.MELT_TEMP
        safe_melt = max(safe_melt, 1.0)
        return IceConvection.LID_FRACTION * safe_melt

    @staticmethod
    def _fallback_interior_temperature(T_melt: float, T_surface: float) -> float:
        """Fallback interior temperature centered within the available thermal window."""
        safe_melt = float(T_melt) if np.isfinite(T_melt) else Thermal.MELT_TEMP
        safe_surface = float(T_surface) if np.isfinite(T_surface) else Thermal.SURFACE_TEMP_MEAN
        return 0.5 * (safe_melt + safe_surface)

    @staticmethod
    def _clip_transition_temperature(T_value: float, T_surface: float, T_melt: float) -> float:
        """Clip transition temperatures to the physically available shell window."""
        safe_surface = float(T_surface) if np.isfinite(T_surface) else Thermal.SURFACE_TEMP_MEAN
        safe_melt = float(T_melt) if np.isfinite(T_melt) else Thermal.MELT_TEMP
        if safe_melt <= safe_surface + 2.0:
            return 0.5 * (safe_melt + safe_surface)
        return float(np.clip(T_value, safe_surface + 1.0, safe_melt - 1.0))

    @staticmethod
    def _collapse_subcritical_state(
            state: ConvectionState,
            total_thickness: float,
            n_nodes: int,
    ) -> ConvectionState:
        """Convert a rheologically warm but subcritical layer into a fully conductive state."""
        if state.is_convecting or state.D_conv <= 0.0:
            return state
        return ConvectionState(
            idx_c=max(n_nodes - 1, 0),
            z_c=total_thickness,
            D_cond=total_thickness,
            D_conv=0.0,
            T_c=state.T_c,
            Ti=state.Ti,
            Ra=state.Ra,
            Nu=1.0,
            is_convecting=False,
            nu_scaling=state.nu_scaling,
        )

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # 0.0 Transition Temperature Calculations
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    @staticmethod
    def howell_cond_base_temp(
            T_melt: float,
            Q_v: float,
    ) -> float:
        """
        Howell (2021) conductive-base temperature derived from rheology.

        T_c = (sqrt(4*T_m*(R/Qv)+1)-1)/(2*(R/Qv))
        T_cond_base = 2*T_c - T_m
        """
        if not np.isfinite(T_melt) or not np.isfinite(Q_v) or Q_v <= 0.0:
            return IceConvection._fallback_howell_transition(T_melt)
        R = Rheology.GAS_CONSTANT
        ratio = R / Q_v
        if not np.isfinite(ratio) or abs(ratio) < 1e-30:
            return IceConvection._fallback_howell_transition(T_melt)
        radicand = 4.0 * T_melt * ratio + 1.0
        if not np.isfinite(radicand) or radicand <= 0.0:
            return IceConvection._fallback_howell_transition(T_melt)
        with np.errstate(invalid='ignore'):
            T_c = (np.sqrt(radicand) - 1.0) / (2.0 * ratio)
        if not np.isfinite(T_c):
            return IceConvection._fallback_howell_transition(T_melt)
        return 2 * T_c - T_melt

    @staticmethod
    def deschamps_interior_temp(
            T_melt: float,
            T_surface: float,
            Q_v: float,
    ) -> float:
        """
        Deschamps & Vilella (2021) Eq. 18: Interior temperature of convecting layer.
        
        Ti = B * (sqrt(1 + (2/B)*(Tm - c2*DTs)) - 1)
        
        Where:
            B = E / (2*R*c1)
            c1 = 1.43, c2 = -0.03 (Deschamps constants)
            DTs = Tm - Ts (temperature drop across shell)
        
        Args:
            T_melt: Melting temperature (K)
            T_surface: Surface temperature (K)
            Q_v: Activation energy (J/mol)
            
        Returns:
            Ti: Interior temperature (K)
        """
        if not np.isfinite(T_melt) or not np.isfinite(T_surface) or not np.isfinite(Q_v) or Q_v <= 0.0:
            return IceConvection._fallback_interior_temperature(T_melt, T_surface)
        R = Rheology.GAS_CONSTANT
        c1 = ConvectionConstants.C1_DESCHAMPS
        c2 = ConvectionConstants.C2_DESCHAMPS
        
        DTs = T_melt - T_surface
        B = Q_v / (2 * R * c1)
        if not np.isfinite(B) or abs(B) < 1e-30:
            return IceConvection._fallback_interior_temperature(T_melt, T_surface)
        radicand = 1.0 + (2.0 / B) * (T_melt - c2 * DTs)
        if not np.isfinite(radicand) or radicand <= 0.0:
            return IceConvection._fallback_interior_temperature(T_melt, T_surface)
        with np.errstate(invalid='ignore'):
            Ti = B * (np.sqrt(radicand) - 1.0)
        if not np.isfinite(Ti):
            return IceConvection._fallback_interior_temperature(T_melt, T_surface)
        return Ti

    @staticmethod
    def green_cond_base_temp(
            T_melt: float,
            T_surface: float,
            Q_v: float,
            eta_ref: float = Rheology.VISCOSITY_REF,
            N: float = 1.0,
            use_composite_transition_closure: bool = False,
            d_grain: Optional[float] = None,
            d_del: Optional[float] = None,
            D0v: Optional[float] = None,
            D0b: Optional[float] = None,
            Q_b: Optional[float] = None,
            p_grain: Optional[float] = None,
    ) -> Tuple[float, float]:
        """
        Green et al. (2021) method for conductive lid base temperature Tc.
        
        Based on Deschamps & Vilella (2021) scaling:
        1. Compute interior temperature Ti from Eq. 18
        2. Compute viscous temperature scale DTv from rheology
        3. Tc = Ti - 2.24 * DTv
        
        Args:
            T_melt: Melting temperature (K)
            T_surface: Surface temperature (K)
            Q_v: Activation energy (J/mol)
            eta_ref: Reference viscosity at Tm (Pa·s)
            N: Stress exponent (1 for Newtonian)
            
        Returns:
            (Tc, Ti): Tuple of lid base temperature and interior temperature (K)
        """
        if (
            not np.isfinite(T_melt)
            or not np.isfinite(T_surface)
            or not np.isfinite(Q_v)
            or Q_v <= 0.0
        ):
            Ti_fallback = IceConvection._clip_transition_temperature(
                IceConvection._fallback_interior_temperature(T_melt, T_surface),
                T_surface,
                T_melt,
            )
            return Ti_fallback, Ti_fallback

        R = Rheology.GAS_CONSTANT
        
        # Step 1: Interior temperature (Deschamps Eq. 18)
        Ti = IceConvection.deschamps_interior_temp(T_melt, T_surface, Q_v)
        
        # Safeguard: Ti must be positive and reasonable
        Ti = IceConvection._clip_transition_temperature(Ti, T_surface, T_melt)
        
        # Step 2: Viscous temperature scale from rheology
        fallback_to_analytic = True
        DTv = 0.0
        
        if use_composite_transition_closure:
            # Use central difference on log(eta) to find viscous temperature scale
            # DTv = -1 / (d ln(eta) / dT)
            dT = 0.1  # K
            eta_plus = IcePhysics.composite_viscosity(
                Ti + dT, d_grain=d_grain, d_del=d_del, D0v=D0v, D0b=D0b,
                Q_diff=Q_v, Q_gbs=Q_b, p_grain=p_grain
            )
            eta_minus = IcePhysics.composite_viscosity(
                Ti - dT, d_grain=d_grain, d_del=d_del, D0v=D0v, D0b=D0b,
                Q_diff=Q_v, Q_gbs=Q_b, p_grain=p_grain
            )
            
            # Safe log difference. If viscosity is clipped (e.g. at 1e12 or 1e25),
            # the derivative will be exactly 0.0, and we must fall back.
            if eta_plus > 0 and eta_minus > 0:
                dlneta_dTi = (np.log(eta_plus) - np.log(eta_minus)) / (2 * dT)
                if np.isfinite(dlneta_dTi) and abs(dlneta_dTi) > 1e-10:
                    DTv = -1.0 / dlneta_dTi
                    fallback_to_analytic = False
        
        if fallback_to_analytic:
            # --- Pitfall #2 guard (DV2021): The FK parameter gamma = Q_v/(R*Ti^2)
            # must be evaluated at Ti (interior temperature), NOT at T_s*T_b.
            # DTv = R*Ti^2 / Q_v = 1/gamma.  The code below computes this via
            # DTv = -eta(Ti) / (d eta/dT |_{Ti}), which is algebraically identical.
            A = Q_v / (N * R * T_melt)
            exponent = np.clip(A * ((T_melt / Ti) - 1), -500, 500)
            exp_term = np.exp(exponent)
            ni = eta_ref * exp_term
            dni_dTi = -eta_ref * (A * T_melt / Ti**2) * exp_term

            if np.abs(dni_dTi) < 1e-100 or not np.isfinite(dni_dTi):
                Tc = IceConvection.howell_cond_base_temp(T_melt, Q_v)
                Tc = IceConvection._clip_transition_temperature(Tc, T_surface, T_melt)
                return Tc, Ti

            DTv = -ni / dni_dTi

            # Verify: DTv should equal R*Ti^2/Q_v for Newtonian (N=1) FK rheology
            DTv_expected = N * R * Ti**2 / Q_v
            assert abs(DTv - DTv_expected) / (abs(DTv_expected) + 1e-30) < 1e-6, (
                f"FK DTv mismatch: computed {DTv:.4f}, expected R*Ti^2/Q_v = {DTv_expected:.4f}. "
                f"Viscous temperature scale may not be evaluated at Ti."
            )
        
        # Step 3: Rheological boundary layer and Tc
        # DTe = 2.24 * DTv (theta_lid parameter)
        theta_lid = ConvectionConstants.THETA_LID
        DTe = theta_lid * DTv
        
        Tc = Ti - DTe
        
        # Ensure physical bounds
        Tc = IceConvection._clip_transition_temperature(Tc, T_surface, T_melt)
        
        return Tc, Ti

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # 1.0 Rayleigh Number
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    @staticmethod
    def rayleigh_number(
            delta_T: float,
            layer_thickness: float,
            T_mean: float,
            use_composite_viscosity: bool = True,
            eta_ref: float = Rheology.VISCOSITY_REF,
            d_grain: Optional[float] = None,
            Q_v: Optional[float] = None,
            Q_b: Optional[float] = None,
            p_grain: Optional[float] = None,
            T_melt: float = Thermal.MELT_TEMP,
    ) -> float:
        """
        Calculates the Rayleigh number (Ra) for a convecting layer.

        Ra = (ρ · g · α · ΔT · d³) / (κ · η)

        Args:
            delta_T: Temperature difference across layer (K)
            layer_thickness: Thickness of convecting layer (m)
            T_mean: Mean temperature of layer (K)
            use_composite_viscosity: Use Howell (2021) composite rheology
            eta_ref: Reference viscosity for simple model (Pa·s)
            d_grain: Ice grain size (m), for composite viscosity
            Q_v: Volume diffusion activation energy (J/mol)
            Q_b: Grain boundary diffusion activation energy (J/mol)
            p_grain: Grain-size exponent (default 2.0)
            T_melt: Basal melting temperature for FK viscosity reference (K)

        Returns:
            Rayleigh number (dimensionless)
        """
        # --- Pitfall #1 guard (DV2021): Ra must use viscosity at the interior/mean
        # convective temperature, NOT at T_surface.  T_mean should be ~ (T_melt+T_c)/2,
        # well above typical surface temperatures (~50-100 K).
        assert T_mean > 150.0, (
            f"Ra viscosity temperature too low ({T_mean:.1f} K); likely using T_surface "
            f"instead of convective-layer mean. Expected T_mean > 150 K."
        )

        # Material properties at mean temperature
        rho = Thermal.density_ice(T_mean)
        k = Thermal.conductivity(T_mean)
        cp = Thermal.specific_heat(T_mean)

        # Viscosity at mean temperature (verified: evaluated at T_mean, not T_surface)
        if use_composite_viscosity:
            eta = IcePhysics.composite_viscosity(
                T_mean,
                d_grain=d_grain,
                Q_diff=Q_v,  # Diffusion creep activation energy
                Q_gbs=Q_b,  # GBS activation energy
                p_grain=p_grain,
            )
        else:
            eta = IcePhysics.viscosity_simple(T_mean, eta_ref, T_melt)

        # Ensure Scaler viscosity
        if hasattr(eta, '__len__'):
            eta = float(eta)

        # Thermal diffusivity: κ = k / (ρ · cp)
        kappa = k / (rho * cp)

        # Rayleigh number
        g = Planetary.GRAVITY
        alpha = ConvectionConstants.ALPHA_EXPANSION

        numerator = rho * g * alpha * delta_T * (layer_thickness ** 3)
        denominator = kappa * eta

        return numerator / denominator

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # 2.0 Nusselt Number
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    @staticmethod
    def nusselt_number(Ra: FloatOrArray, simple: bool = True) -> FloatOrArray:
        """
        Calculates Nusselt number (Nu) using stagnant-lid scaling.

        For stagnant-lid convection (Solomatov & Moresi 2000, Green et al. 2021):
            Nu = C · Ra^xi, where xi = 1/3 and C = 0.3446

        Nu represents heat transfer enhancement over pure conduction:
            q_conv = Nu · q_cond

        Args:
            Ra: Rayleigh number (dimensionless)
            simple: Use simple scaling (True) or return raw Nu without minimum (False)

        Returns:
            Nusselt number (dimensionless, >= 1.0)
        """
        Ra_crit = ConvectionConstants.RA_CRIT
        prefactor = ConvectionConstants.NU_PREFACTOR  # 0.3446 (Green et al.)
        xi = ConvectionConstants.BETA_NU  # 1/3

        if np.isscalar(Ra):
            if Ra < Ra_crit:
                return 1.0  # Subcritical: conduction only
            Nu = prefactor * Ra ** xi
            return max(Nu, 1.0) if simple else Nu
        else:
            ra_arr = np.asarray(Ra)
            Nu = np.ones_like(ra_arr, dtype=float)
            supercritical = ra_arr >= Ra_crit
            Nu[supercritical] = prefactor * ra_arr[supercritical] ** xi
            if simple:
                Nu = np.maximum(Nu, 1.0)
            return Nu

    @staticmethod
    def nusselt_number_green(
            Ra: float,
            Ti: float,
            Tc: float,
            DT: float,
    ) -> float:
        """
        Green et al. (2021) Nusselt number with internal heating correction.
        
        Qt = (k*DT/b) * C * Ra^xi * ((Ti-Tc)/DT)^zeta
        
        This means effective Nu = C * Ra^xi * ((Ti-Tc)/DT)^zeta
        
        Args:
            Ra: Rayleigh number
            Ti: Interior temperature (K)
            Tc: Lid base temperature (K)
            DT: Temperature drop across convective layer = Tm - Tc (K)
            
        Returns:
            Nu: Nusselt number with internal heating correction
        """
        Ra_crit = ConvectionConstants.RA_CRIT
        C = ConvectionConstants.NU_PREFACTOR  # 0.3446
        xi = ConvectionConstants.BETA_NU  # 1/3
        zeta = ConvectionConstants.ZETA_NU  # 4/3
        
        if Ra < Ra_crit or DT <= 0:
            return 1.0
        
        # Internal heating correction factor
        temp_ratio = (Ti - Tc) / DT if DT > 0 else 1.0
        temp_ratio = max(temp_ratio, 0.01)  # Prevent negative/zero
        
        Nu = C * (Ra ** xi) * (temp_ratio ** zeta)
        
        return max(Nu, 1.0)

    @staticmethod
    def nusselt_number_isoviscous(Ra: FloatOrArray) -> FloatOrArray:
        """
        Isoviscous Nu-Ra scaling: Nu = 0.088 · Ra^0.28

        Diagnostic only — Europa's viscosity contrast Δη ~ 1e7 violates
        the isoviscous assumption.  Do not use for production runs.

        Reference: compilation in Solomatov (1995), Table 1.
        """
        Ra_crit = ConvectionConstants.RA_CRIT
        if np.isscalar(Ra):
            if Ra < Ra_crit:
                return 1.0
            return max(0.088 * Ra ** 0.28, 1.0)
        ra_arr = np.asarray(Ra)
        Nu = np.ones_like(ra_arr, dtype=float)
        supercritical = ra_arr >= Ra_crit
        Nu[supercritical] = 0.088 * ra_arr[supercritical] ** 0.28
        return np.maximum(Nu, 1.0)

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # 2.5 Deschamps & Vilella (2021) Mixed-Heating Scaling
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    @staticmethod
    def dv2021_interior_temperature(
            Ra_surf: float,
            gamma: float,
            H_tilde: float,
            f: float = 1.0,
            ur_regime: str = "lt1",
    ) -> float:
        """
        Solve Eq. 21 from Deschamps & Vilella (2021) for nondimensional
        interior temperature T̃_m using Brent's method.

        Args:
            Ra_surf: Surface Rayleigh number (uses surface viscosity)
            gamma: ln(Delta_eta) = E_a * DT / (R * T_m^2) for FK rheology
            H_tilde: Nondimensional internal heating = rho*H*D^2 / (k*DT)
            f: Geometry factor (1.0 = Cartesian / thin shell limit)
            ur_regime: "lt1" (Ur < 1, bottom-heated dominant) or
                       "gt1" (Ur > 1, internally-heated dominant)

        Returns:
            T_m_tilde: Nondimensional interior temperature
        """
        # Table 2 coefficients
        if ur_regime == "lt1":
            a1, c1, c2, c3, c4 = 1.23, 3.5, -2.3, 0.25, 1.0
        else:  # "gt1"
            a1, c1, c2, c3, c4 = 1.23, 4.4, -3.0, 1.0 / 3.0, 1.72

        geom = (1 + f + f**2) / 3.0
        H_term = (H_tilde * geom) ** c4 if H_tilde > 0 else 0.0

        def residual(Tm):
            Ra_eff = Ra_surf * np.exp(gamma * Tm)
            rhs = 1.0 - a1 / (f**2 * gamma)
            if H_term > 0:
                rhs += (c1 + c2 * f) * H_term / Ra_eff**c3
            return Tm - rhs

        # For H_tilde=0 the residual is linear: Tm - (1 - a1/(f^2*gamma)),
        # with root at Tm = 1 - a1/(f^2*gamma).  When gamma is small
        # (< ~2.5), this root can fall below 0.5.  Use a wider bracket
        # to handle all physically valid gamma values.
        return brentq(residual, 0.0, 2.0, xtol=1e-10)

    @staticmethod
    def dv2021_surface_heat_flux(
            Ra_eff: float,
            gamma: float,
            ur_regime: str = "lt1",
    ) -> float:
        """
        Eq. 23 from Deschamps & Vilella (2021): nondimensional surface heat flux.

        Phi_top = a * Ra_eff^b / gamma^c

        Args:
            Ra_eff: Effective (interior) Rayleigh number
            gamma: Viscosity contrast parameter
            ur_regime: "lt1" or "gt1"

        Returns:
            Phi_top: Nondimensional surface heat flux
        """
        if ur_regime == "lt1":
            a, b, c = 1.46, 0.27, 1.21
        else:
            a, b, c = 1.57, 0.27, 1.21

        Phi_top = a * Ra_eff**b / gamma**c

        # FK correction: empirical factor for FK vs. full Arrhenius mismatch
        if ConfigManager.get("convection", "FK_CORRECTION", False):
            factor = ConfigManager.get("convection", "FK_CORRECTION_FACTOR", 0.75)
            Phi_top *= factor

        return Phi_top

    @staticmethod
    def dv2021_lid_thickness(
            Ra_eff: float,
            gamma: float,
            ur_regime: str = "lt1",
    ) -> float:
        """
        Eq. 26 from Deschamps & Vilella (2021): nondimensional stagnant lid thickness.

        d_lid = a_lid * gamma^c / Ra_eff^b

        Args:
            Ra_eff: Effective (interior) Rayleigh number
            gamma: Viscosity contrast parameter
            ur_regime: "lt1" or "gt1"

        Returns:
            d_lid_nd: Nondimensional lid thickness (fraction of shell)
        """
        if ur_regime == "lt1":
            a_lid, b, c = 0.633, 0.27, 1.21
        else:
            a_lid, b, c = 0.667, 0.27, 1.21

        return a_lid * gamma**c / Ra_eff**b

    @staticmethod
    def dv2021_solve(
            Ra_surf: float,
            gamma: float,
            H_tilde: float,
            f: float = 1.0,
    ) -> dict:
        """
        Full DV2021 solve with automatic Urey-ratio regime switching.

        Algorithm (Section 5 of DV2021):
        1. Solve with Ur<1 coefficients -> compute Phi_bot from Eq. 11
        2. If Phi_bot < 0: re-solve with Ur>1 coefficients
        3. If new Phi_bot > 0 (boundary contradiction): set Phi_bot=0, recalculate

        Eq. 11 (general f): Phi_bot = f^2 * Phi_top - (1+f+f^2)/3 * H_tilde

        Args:
            Ra_surf: Surface Rayleigh number
            gamma: Viscosity contrast parameter
            H_tilde: Nondimensional internal heating
            f: Geometry factor (1.0 for Cartesian / thin shell)

        Returns:
            dict with keys: T_m_tilde, Ra_eff, Phi_top, Phi_bot,
                            d_lid_nd, regime
        """
        geom = (1 + f + f**2) / 3.0

        # Step 1: try Ur<1 (bottom-heated dominant)
        T_m = IceConvection.dv2021_interior_temperature(
            Ra_surf, gamma, H_tilde, f=f, ur_regime="lt1"
        )
        Ra_eff = Ra_surf * np.exp(gamma * T_m)
        Phi_top = IceConvection.dv2021_surface_heat_flux(Ra_eff, gamma, ur_regime="lt1")
        Phi_bot = f**2 * Phi_top - geom * H_tilde
        d_lid = IceConvection.dv2021_lid_thickness(Ra_eff, gamma, ur_regime="lt1")
        regime = "lt1"

        # Step 2: if Phi_bot < 0, internal heating dominates -> switch to Ur>1
        if Phi_bot < 0:
            T_m = IceConvection.dv2021_interior_temperature(
                Ra_surf, gamma, H_tilde, f=f, ur_regime="gt1"
            )
            Ra_eff = Ra_surf * np.exp(gamma * T_m)
            Phi_top = IceConvection.dv2021_surface_heat_flux(Ra_eff, gamma, ur_regime="gt1")
            Phi_bot = f**2 * Phi_top - geom * H_tilde
            d_lid = IceConvection.dv2021_lid_thickness(Ra_eff, gamma, ur_regime="gt1")
            regime = "gt1"

            # Step 3: contradiction — Phi_bot should be <= 0 for Ur>1
            if Phi_bot > 0:
                Phi_bot = 0.0
                # Recalculate Phi_top from energy balance: Phi_top = geom * H_tilde / f^2
                Phi_top = geom * H_tilde / f**2

        return {
            "T_m_tilde": T_m,
            "Ra_eff": Ra_eff,
            "Phi_top": Phi_top,
            "Phi_bot": Phi_bot,
            "d_lid_nd": d_lid,
            "regime": regime,
        }

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # 3.0 Conductive lid thickness
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    @staticmethod
    def lid_thickness(
            total_thickness: float,
            T_surface: float = Thermal.SURFACE_TEMP_MEAN,
            T_melt: float = Thermal.MELT_TEMP,
            lid_fraction: Optional[float] = None,
            Q_v: Optional[float] = None,
    ) -> float:
        """
        Estimates the thickness of the conductive lid (z_c).

        The lid extends from surface to the rheological transition temperature,
        where viscosity drops enough to permit convective flow.

        Args:
            total_thickness: Total ice shell thickness (m)
            T_surface: Surface temperature (K)
            T_melt: Basal melting temperature (K)
            lid_fraction: T_transition / T_melt ratio (default LID_FRACTION)

        Returns:
            Conductive lid thickness (m)
        """

        if Q_v is not None:
            T_cond_base = IceConvection.howell_cond_base_temp(T_melt, Q_v)
            T_cond_base = np.clip(T_cond_base, T_surface + 1e-6, T_melt - 1e-6)
            fraction = (T_cond_base - T_surface) / (T_melt - T_surface)
            z_c = fraction * total_thickness
        else:
            if lid_fraction is None:
                lid_fraction = IceConvection.LID_FRACTION
            T_transition = lid_fraction * T_melt
            # Linear estimate based on temperature profile
            fraction = (T_transition - T_surface) / (T_melt - T_surface)
            z_c = fraction * total_thickness


        return max(z_c, 0.0)

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # 3.1 Ra-criterion based convective boundary
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    @staticmethod
    def find_convective_boundary(
            total_thickness: float,
            T_surface: float = Thermal.SURFACE_TEMP_MEAN,
            T_melt: float = Thermal.MELT_TEMP,
            use_composite_viscosity: bool = True,
            eta_ref: float = Rheology.VISCOSITY_REF,
            d_grain: Optional[float] = None,
            Q_v: Optional[float] = None,
            Q_b: Optional[float] = None,
            p_grain: Optional[float] = None,
            n_search: int = 50,
    ) -> float:
        """
        Find lid thickness where Ra first exceeds Ra_crit.

        Searches from base upward through candidate convective layer thicknesses
        to find where the Rayleigh number first exceeds the critical value.
        This determines the convective boundary from first principles rather
        than using a prescribed temperature fraction.

        Args:
            total_thickness: Total ice shell thickness (m)
            T_surface: Surface temperature (K)
            T_melt: Basal melting temperature (K)
            use_composite_viscosity: Use Howell (2021) composite rheology
            eta_ref: Reference viscosity for simple model (Pa·s)
            d_grain: Ice grain size (m), for composite viscosity
            Q_v: Volume diffusion activation energy (J/mol)
            Q_b: Grain boundary diffusion activation energy (J/mol)
            n_search: Number of candidate layers to test

        Returns:
            z_c: Conductive lid thickness (m). If no convection possible,
                 returns total_thickness (pure conduction).
        """
        Ra_crit = ConvectionConstants.RA_CRIT

        # Search from thin convective layer to thick (base upward)
        for i in range(1, n_search):
            # Candidate convective thickness (measured from base)
            h_conv = (i / n_search) * total_thickness
            z_c_candidate = total_thickness - h_conv

            # Temperature at candidate lid base (linear profile approximation)
            frac = z_c_candidate / total_thickness
            T_lid_base = T_surface + frac * (T_melt - T_surface)

            # Ensure physical temperature bounds
            T_lid_base = np.clip(T_lid_base, T_surface + 1e-6, T_melt - 1e-6)

            # Properties for this candidate sublayer
            delta_T = T_melt - T_lid_base
            T_mean = (T_melt + T_lid_base) / 2

            # Calculate Ra for this candidate convective layer
            Ra = IceConvection.rayleigh_number(
                delta_T, h_conv, T_mean,
                use_composite_viscosity, eta_ref,
                d_grain=d_grain, Q_v=Q_v, Q_b=Q_b, p_grain=p_grain,
                T_melt=T_melt,
            )

            # First layer where Ra >= Ra_crit defines the boundary
            if Ra >= Ra_crit:
                return z_c_candidate

        # No convection possible - entire shell is conductive
        return total_thickness

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # 4.0 Effective conductivity
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    @staticmethod
    def effective_conductivity(
            T: FloatOrArray,
            z: FloatOrArray,
            total_thickness: float,
            T_surface: float = Thermal.SURFACE_TEMP_MEAN,
            T_melt: float = Thermal.MELT_TEMP,
            use_composite_viscosity: bool = True,
            eta_ref: float = Rheology.VISCOSITY_REF,
            d_grain: Optional[float] = None,
            Q_v: Optional[float] = None,
            Q_b: Optional[float] = None,
            p_grain: Optional[float] = None,
    ) -> FloatOrArray:
        """
        Returns effective thermal conductivity accounting for convection.

        Uses Howell (2021) rheological transition temperature to determine
        the convective boundary:
        - Conductive lid (z < z_c): k_eff = k(T)
        - Convective sublayer (z >= z_c): k_eff = Nu · k(T)

        The lid thickness is determined by where the temperature reaches
        T_cond_base (the rheological transition temperature), not by Ra_crit.

        Args:
            T: Temperature at node(s) (K)
            z: Depth of node(s) (m)
            total_thickness: Total ice shell thickness (m)
            T_surface: Surface temperature (K)
            T_melt: Basal temperature (K)
            use_composite_viscosity: Use Howell (2021) rheology
            eta_ref: Reference viscosity for simple model (Pa·s)
            d_grain: Ice grain size (m), for composite viscosity
            Q_v: Volume diffusion activation energy (J/mol)
            Q_b: Grain boundary diffusion activation energy (J/mol)

        Returns:
            Effective thermal conductivity (W/m·K)
        """
        z_scalar = np.isscalar(z)
        T_arr = np.asarray(T)
        z_arr = np.asarray(z)

        # Base conductivity
        k = Thermal.conductivity(T_arr)

        # Calculate lid thickness using Howell transition temperature (if Q_v provided)
        z_c = IceConvection.lid_thickness(
            total_thickness, T_surface, T_melt, Q_v=Q_v
        )
        convective_thickness = total_thickness - z_c

        if convective_thickness <= 0.0:
            return k  # Shell too thin for convection

        # Convective sublayer properties
        if Q_v is not None:
            T_cond_base = IceConvection.howell_cond_base_temp(T_melt, Q_v)
            T_cond_base = np.clip(T_cond_base, T_surface + 1e-6, T_melt - 1e-6)
            delta_T = T_melt - T_cond_base
            T_mean = (T_melt + T_cond_base) / 2
        else:
            T_transition = IceConvection.LID_FRACTION * T_melt
            delta_T = T_melt - T_transition
            T_mean = (T_melt + T_transition) / 2

        # Rayleigh and Nusselt numbers
        Ra = IceConvection.rayleigh_number(
            delta_T, convective_thickness, T_mean,
            use_composite_viscosity, eta_ref,
            d_grain=d_grain, Q_v=Q_v, Q_b=Q_b, p_grain=p_grain,
            T_melt=T_melt,
        )
        Nu = IceConvection.nusselt_number(Ra)

        # Apply Nu enhancement only in convective layer
        if z_scalar:
            if float(z_arr) > z_c:
                return float(k * Nu)
            return float(k)
        else:
            k_eff = np.copy(k)
            convective_mask = z_arr > z_c
            k_eff[convective_mask] *= Nu
            return k_eff

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # 5.0 Convection State Queries
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    @staticmethod
    def is_convecting(
            total_thickness: float,
            T_surface: float = Thermal.SURFACE_TEMP_MEAN,
            T_melt: float = Thermal.MELT_TEMP,
            use_composite_viscosity: bool = True,
            eta_ref: float = Rheology.VISCOSITY_REF,
            Q_v: Optional[float] = None,
            Q_b: Optional[float] = None,
            d_grain: Optional[float] = None,
            p_grain: Optional[float] = None,
            use_composite_transition_closure: bool = False,
            d_del: Optional[float] = None,
            D0v: Optional[float] = None,
            D0b: Optional[float] = None,
    ) -> Tuple[bool, float]:
        """
        Determines if the ice shell is in a convective regime.

        Convection initiates when Ra > Ra_critical for the convective sublayer.

        Args:
            total_thickness: Ice shell thickness (m)
            T_surface: Surface temperature (K)
            T_melt: Basal temperature (K)
            use_composite_viscosity: Use Howell (2021) rheology
            eta_ref: Reference viscosity for simple model (Pa·s)
            Q_v: Volume diffusion activation energy (J/mol)
            Q_b: Grain boundary diffusion activation energy (J/mol)
            d_grain: Ice grain size (m), for composite viscosity

        Returns:
            (is_convecting, Ra): Tuple of convection state and Rayleigh number
        """
        T_cond_base, _ = IceConvection.compute_transition_temperature(
            T_melt=T_melt,
            T_surface=T_surface,
            Q_v=Q_v,
            eta_ref=eta_ref,
            nu_scaling=ConvectionConstants.NU_SCALING,
            use_composite_transition_closure=use_composite_transition_closure,
            d_grain=d_grain,
            d_del=d_del,
            D0v=D0v,
            D0b=D0b,
            Q_b=Q_b,
            p_grain=p_grain,
        )
        if T_melt <= T_surface:
            return False, 0.0

        fraction = (T_cond_base - T_surface) / (T_melt - T_surface)
        z_c = float(np.clip(fraction, 0.0, 1.0) * total_thickness)
        convective_thickness = total_thickness - z_c

        if convective_thickness <= 0:
            return False, 0.0

        delta_T = T_melt - T_cond_base
        T_mean = (T_melt + T_cond_base) / 2

        Ra = IceConvection.rayleigh_number(
            delta_T, convective_thickness, T_mean,
            use_composite_viscosity, eta_ref,
            d_grain=d_grain, Q_v=Q_v, Q_b=Q_b, p_grain=p_grain,
            T_melt=T_melt,
        )

        return Ra >= ConvectionConstants.RA_CRIT, Ra

    @staticmethod
    def get_diagnostics(
            total_thickness: float,
            T_surface: float = Thermal.SURFACE_TEMP_MEAN,
            T_melt: float = Thermal.MELT_TEMP,
            use_composite_viscosity: bool = True,
            eta_ref: float = Rheology.VISCOSITY_REF,
            Q_v: Optional[float] = None,
            Q_b: Optional[float] = None,
            d_grain: Optional[float] = None,
            p_grain: Optional[float] = None,
            use_composite_transition_closure: bool = False,
            d_del: Optional[float] = None,
            D0v: Optional[float] = None,
            D0b: Optional[float] = None,
    ) -> dict:
        """
        Returns a summary dictionary of convection parameters.

        Useful for diagnostics and logging.

        Returns:
            Dictionary with thickness, Ra, Nu, and convection state
        """
        T_cond_base, _ = IceConvection.compute_transition_temperature(
            T_melt=T_melt,
            T_surface=T_surface,
            Q_v=Q_v,
            eta_ref=eta_ref,
            nu_scaling=ConvectionConstants.NU_SCALING,
            use_composite_transition_closure=use_composite_transition_closure,
            d_grain=d_grain,
            d_del=d_del,
            D0v=D0v,
            D0b=D0b,
            Q_b=Q_b,
            p_grain=p_grain,
        )
        if T_melt > T_surface:
            z_c = float(np.clip((T_cond_base - T_surface) / (T_melt - T_surface), 0.0, 1.0) * total_thickness)
        else:
            z_c = total_thickness
        convecting, Ra = IceConvection.is_convecting(
            total_thickness, T_surface, T_melt,
            use_composite_viscosity, eta_ref,
            Q_v=Q_v, Q_b=Q_b, d_grain=d_grain, p_grain=p_grain,
            use_composite_transition_closure=use_composite_transition_closure,
            d_del=d_del, D0v=D0v, D0b=D0b,
        )
        Nu = IceConvection.nusselt_number(Ra)

        return {
            'total_thickness_km': total_thickness / 1000,
            'conductive_lid_km': z_c / 1000,
            'convective_layer_km': (total_thickness - z_c) / 1000,
            'is_convecting': convecting,
            'rayleigh_number': Ra,
            'nusselt_number': Nu,
            'critical_rayleigh': ConvectionConstants.RA_CRIT,
        }

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # 6.0 PHASE 2: Temperature-Profile Based Convection (Green et al. 2021)
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    @staticmethod
    def compute_transition_temperature(
            T_melt: float = Thermal.MELT_TEMP,
            T_surface: float = Thermal.SURFACE_TEMP_MEAN,
            Q_v: Optional[float] = None,
            eta_ref: float = Rheology.VISCOSITY_REF,
            default_T_c: float = 250.0,
            nu_scaling: str = ConvectionConstants.NU_SCALING,
            use_composite_transition_closure: bool = False,
            d_grain: Optional[float] = None,
            d_del: Optional[float] = None,
            D0v: Optional[float] = None,
            D0b: Optional[float] = None,
            Q_b: Optional[float] = None,
            p_grain: Optional[float] = None,
    ) -> Tuple[float, float]:
        """
        Computes the rheological transition temperature T_c and interior temperature Ti.

        Below T_c, viscosity is too high for convective flow (stagnant lid).
        Above T_c, the ice is ductile enough to convect.

        Args:
            T_melt: Basal melting temperature (K)
            T_surface: Surface temperature (K)
            Q_v: Volume diffusion activation energy (J/mol)
            eta_ref: Reference viscosity at Tm (Pa·s)
            default_T_c: Default transition temp if Q_v not provided (K)
            nu_scaling: Scaling law selector ("green", "howell", "isoviscous_benchmark")

        Returns:
            (T_c, Ti): Critical transition temperature and interior temperature (K)
        """
        if not np.isfinite(T_melt) or not np.isfinite(T_surface) or T_melt <= T_surface + 1e-6:
            safe_transition = IceConvection._clip_transition_temperature(
                IceConvection._fallback_interior_temperature(T_melt, T_surface),
                T_surface,
                T_melt,
            )
            return safe_transition, safe_transition
        if Q_v is not None:
            if nu_scaling == "howell":
                # Howell (2021) method
                T_c = IceConvection.howell_cond_base_temp(T_melt, Q_v)
                T_c = IceConvection._clip_transition_temperature(
                    max(T_c, 180.0),
                    T_surface,
                    T_melt,
                )
                Ti = IceConvection._clip_transition_temperature((T_melt + T_c) / 2, T_surface, T_melt)
            elif nu_scaling in ("green", "isoviscous_benchmark"):
                T_c, Ti = IceConvection.green_cond_base_temp(
                    T_melt, T_surface, Q_v, eta_ref,
                    use_composite_transition_closure=use_composite_transition_closure,
                    d_grain=d_grain, d_del=d_del, D0v=D0v, D0b=D0b,
                    Q_b=Q_b, p_grain=p_grain
                )
            elif nu_scaling == "dv2021":
                # DV2021 uses the same Deschamps Eq.18 interior temperature as Green,
                # but with FK viscosity (not composite). Tc follows from theta_lid.
                T_c, Ti = IceConvection.green_cond_base_temp(
                    T_melt, T_surface, Q_v, eta_ref,
                    use_composite_transition_closure=False,
                    d_grain=d_grain, d_del=d_del, D0v=D0v, D0b=D0b,
                    Q_b=Q_b, p_grain=p_grain,
                )
            else:
                raise ValueError(f"Unknown nu_scaling={nu_scaling!r}")
            return T_c, Ti
        else:
            # Default values
            T_c = IceConvection._clip_transition_temperature(default_T_c, T_surface, T_melt)
            Ti = IceConvection._clip_transition_temperature((T_melt + T_c) / 2, T_surface, T_melt)
            return T_c, Ti

    @staticmethod
    def scan_temperature_profile(
            T_profile: npt.NDArray[np.float64],
            z_grid: npt.NDArray[np.float64],
            total_thickness: float,
            T_melt: float = Thermal.MELT_TEMP,
            Q_v: Optional[float] = None,
            Q_b: Optional[float] = None,
            d_grain: Optional[float] = None,
            p_grain: Optional[float] = None,
            use_composite_viscosity: bool = True,
            eta_ref: float = Rheology.VISCOSITY_REF,
            use_composite_transition_closure: bool = False,
            d_del: Optional[float] = None,
            D0v: Optional[float] = None,
            D0b: Optional[float] = None,
            nu_scaling: Optional[str] = None,
    ) -> ConvectionState:
        """
        Scans the current temperature profile to locate the conductive/convective interface.

        This is the core Phase 2 algorithm (Green et al. 2021, Deschamps & Vilella 2021):
        1. Determine critical temperature T_c from rheology (Deschamps method)
        2. Find depth index where T(z) >= T_c
        3. Calculate Ra and Nu for the convective sublayer

        The viscosity model used for Ra is determined by NU_SCALING so that Ra
        and Nu are computed under the same closure the scaling law was calibrated
        against (e.g. Green → FK viscosity, not composite).

        Args:
            T_profile: Current temperature array from surface to base (K)
            z_grid: Depth coordinates (m), same length as T_profile
            total_thickness: Total ice shell thickness (m)
            T_melt: Basal melting temperature (K)
            Q_v: Volume diffusion activation energy (J/mol)
            Q_b: Grain boundary diffusion activation energy (J/mol)
            d_grain: Ice grain size (m)
            use_composite_viscosity: Use Howell (2021) composite rheology
            eta_ref: Reference viscosity for simple model (Pa·s)
            nu_scaling: Override scaling law selector (default: read from config)

        Returns:
            ConvectionState: Complete convection diagnostics
        """
        N_nodes = len(T_profile)
        if N_nodes == 0 or len(z_grid) != N_nodes:
            raise ValueError("Temperature profile and depth grid must be non-empty and the same length.")
        if (
            not np.all(np.isfinite(T_profile))
            or not np.all(np.isfinite(z_grid))
            or not np.isfinite(total_thickness)
            or total_thickness <= 0.0
            or not np.isfinite(T_melt)
        ):
            raise ValueError("Non-finite or non-physical inputs reached the convection profile scan.")

        T_surface = float(T_profile[0])
        nu_scaling = nu_scaling if nu_scaling is not None else ConvectionConstants.NU_SCALING

        if T_melt <= T_surface + 1e-6:
            return ConvectionState(
                idx_c=N_nodes - 1,
                z_c=total_thickness,
                D_cond=total_thickness,
                D_conv=0.0,
                T_c=T_surface,
                Ti=T_surface,
                Ra=0.0,
                Nu=1.0,
                is_convecting=False,
                nu_scaling=nu_scaling,
            )

        # Step A: Determine transition temperature
        T_c, Ti = IceConvection.compute_transition_temperature(
            T_melt, T_surface, Q_v, eta_ref, nu_scaling=nu_scaling,
            use_composite_transition_closure=use_composite_transition_closure,
            d_grain=d_grain, d_del=d_del, D0v=D0v, D0b=D0b,
            Q_b=Q_b, p_grain=p_grain
        )
        
        # Step B: Scan profile to find interface index
        # Find first index where T >= T_c (measured from surface)
        warm_indices = np.where(T_profile >= T_c)[0]
        
        if len(warm_indices) == 0:
            # Shell is too cold to convect anywhere
            return ConvectionState(
                idx_c=N_nodes - 1,
                z_c=total_thickness,
                D_cond=total_thickness,
                D_conv=0.0,
                T_c=T_c,
                Ti=Ti,
                Ra=0.0,
                Nu=1.0,
                is_convecting=False,
                nu_scaling=nu_scaling,
            )
        
        idx_c = warm_indices[0]  # First index where convection possible
        
        if 0 < idx_c < N_nodes:
            # Linearly interpolate to find exact depth where T == T_c
            T_above = T_profile[idx_c - 1]
            T_below = T_profile[idx_c]
            z_above = z_grid[idx_c - 1]
            z_below = z_grid[idx_c]
            
            # Protect against division by zero (though T_below > T_above due to gradient)
            if T_below > T_above:
                frac = (T_c - T_above) / (T_below - T_above)
                z_c = z_above + frac * (z_below - z_above)
            else:
                z_c = z_grid[idx_c]
        else:
            z_c = z_grid[idx_c] if idx_c < N_nodes else total_thickness
        
        D_cond = z_c
        D_conv = total_thickness - z_c
        
        # Step C: Calculate Rayleigh and Nusselt numbers for convective layer
        if D_conv <= 0:
            return ConvectionState(
                idx_c=idx_c,
                z_c=z_c,
                D_cond=D_cond,
                D_conv=0.0,
                T_c=T_c,
                Ti=Ti,
                Ra=0.0,
                Nu=1.0,
                is_convecting=False,
                nu_scaling=nu_scaling,
            )
        
        # Temperature difference across convective layer (Green et al.: DT = Tm - Tc)
        DT = T_melt - T_c
        T_mean = (T_melt + T_c) / 2

        # Override viscosity model for Ra to match scaling-law calibration basis.
        # Green et al. (2021) calibrated C·Ra^(1/3) under FK viscosity;
        # feeding composite-viscosity Ra into that law mixes closures.
        if nu_scaling == "green":
            ra_use_composite = False  # FK viscosity
        elif nu_scaling == "isoviscous_benchmark":
            ra_use_composite = False  # FK viscosity
        elif nu_scaling == "howell":
            ra_use_composite = use_composite_viscosity
        elif nu_scaling == "dv2021":
            ra_use_composite = False  # DV2021 calibrated on FK viscosity
        else:
            raise ValueError(f"Unknown nu_scaling={nu_scaling!r}")

        Ra = IceConvection.rayleigh_number(
            DT, D_conv, T_mean,
            ra_use_composite, eta_ref,
            d_grain=d_grain, Q_v=Q_v, Q_b=Q_b, p_grain=p_grain,
            T_melt=T_melt,
        )

        # Step D: Nusselt number from the matched scaling law
        if nu_scaling == "green":
            Nu = IceConvection.nusselt_number_green(Ra, Ti, T_c, DT)
        elif nu_scaling == "isoviscous_benchmark":
            Nu = IceConvection.nusselt_number_isoviscous(Ra)
        elif nu_scaling == "howell":
            Nu = IceConvection.nusselt_number(Ra)
        elif nu_scaling == "dv2021":
            # DV2021 scaling: compute Ra_surf (at T_c, not T_mean) and gamma,
            # then use dv2021_solve for Phi_top which serves as effective Nu.
            R = Rheology.GAS_CONSTANT
            Q_v_use = Q_v if Q_v is not None else Rheology.ACTIVATION_ENERGY_V

            # Compute surface (lid-base) viscosity for Ra_surf
            eta_surf = IcePhysics.viscosity_simple(T_c, eta_ref, T_melt)
            rho = Thermal.density_ice(T_mean)
            k = Thermal.conductivity(T_mean)
            cp = Thermal.specific_heat(T_mean)
            kappa = k / (rho * cp)
            g = Planetary.GRAVITY
            alpha = ConvectionConstants.ALPHA_EXPANSION

            Ra_surf = rho * g * alpha * DT * D_conv**3 / (kappa * eta_surf)

            # gamma = ln(Delta_eta) = Q/(R) * (1/T_c - 1/T_melt) for FK rheology
            gamma = Q_v_use / R * (1.0 / T_c - 1.0 / T_melt)
            gamma = max(gamma, 1.0)  # DV2021 requires gamma >= ~1

            # H_tilde = rho * H_vol * D^2 / (k * DT) — default 0 (bottom-heated)
            # Internal heating is handled externally by the tidal model, so set 0 here.
            H_tilde = 0.0

            # Geometry correction from config (1.0 = Cartesian thin-shell limit)
            f = ConfigManager.get("convection", "GEOMETRY_CORRECTION", 1.0)

            dv = IceConvection.dv2021_solve(Ra_surf, gamma, H_tilde, f=f)
            Nu = max(dv["Phi_top"], 1.0)
        else:
            raise ValueError(f"Unknown nu_scaling={nu_scaling!r}")

        is_convecting = Ra >= ConvectionConstants.RA_CRIT

        return ConvectionState(
            idx_c=idx_c,
            z_c=z_c,
            D_cond=D_cond,
            D_conv=D_conv,
            T_c=T_c,
            Ti=Ti,
            Ra=Ra,
            Nu=Nu,
            is_convecting=is_convecting,
            nu_scaling=nu_scaling,
        )

    @staticmethod
    def build_conductivity_profile(
            T_profile: npt.NDArray[np.float64],
            z_grid: npt.NDArray[np.float64],
            total_thickness: float,
            T_melt: float = Thermal.MELT_TEMP,
            Q_v: Optional[float] = None,
            Q_b: Optional[float] = None,
            d_grain: Optional[float] = None,
            p_grain: Optional[float] = None,
            use_composite_viscosity: bool = True,
            eta_ref: float = Rheology.VISCOSITY_REF,
            use_composite_transition_closure: bool = False,
            d_del: Optional[float] = None,
            D0v: Optional[float] = None,
            D0b: Optional[float] = None,
            porosity: float = 0.0,
            salt_fraction: float = 0.0,
            salt_scaling_factor: float = 1.0,
            porosity_cure_temp: Optional[float] = None,
            nu_ramp_factor: float = 1.0,
            use_onset_consistent_partition: bool = False,
            convection_adjuster=None,
            q_ocean: float = 0.0,
    ) -> Tuple[npt.NDArray[np.float64], ConvectionState]:
        """
        Builds the spatially-varying conductivity profile k(z) with convection enhancement.
        
        Phase 2 Implementation (Green et al. 2021):
        - Conductive lid (z < z_c): k = k_ice(T)
        - Convective interior (z >= z_c): k = Nu * k_ice(T)
        
        Args:
            T_profile: Current temperature array (K)
            z_grid: Depth coordinates (m)
            total_thickness: Total shell thickness (m)
            T_melt: Basal melting temperature (K)
            Q_v, Q_b: Activation energies (J/mol)
            d_grain: Grain size (m)
            use_composite_viscosity: Use composite rheology
            eta_ref: Reference viscosity (Pa·s)
            porosity: Porosity fraction
            salt_fraction: Salt fraction
            salt_scaling_factor: Salt conductivity factor B_k
            porosity_cure_temp: Temperature above which porosity disappears
            
        Returns:
            (k_profile, convection_state): Conductivity array and diagnostics
        """
        N_nodes = len(T_profile)
        T_surface = T_profile[0]
        nu_scaling = ConvectionConstants.NU_SCALING

        if nu_scaling == "howell":
            # Pre-Phase 2 / Howell (2021) method: Estimate z_c from linear assumptions
            z_c = IceConvection.lid_thickness(
                total_thickness, T_surface, T_melt, Q_v=Q_v
            )
            D_cond = z_c
            D_conv = total_thickness - z_c

            if Q_v is not None:
                T_c = np.clip(IceConvection.howell_cond_base_temp(T_melt, Q_v), T_surface + 1e-6, T_melt - 1e-6)
                delta_T = T_melt - T_c
                T_mean = (T_melt + T_c) / 2
            else:
                T_transition = IceConvection.LID_FRACTION * T_melt
                delta_T = T_melt - T_transition
                T_mean = (T_melt + T_transition) / 2
                T_c = T_transition

            is_convecting = False
            Ra = 0.0
            Nu = 1.0
            idx_c = N_nodes - 1

            if D_conv > 0:
                Ra = IceConvection.rayleigh_number(
                    delta_T, D_conv, T_mean,
                    use_composite_viscosity, eta_ref,
                    d_grain=d_grain, Q_v=Q_v, Q_b=Q_b, p_grain=p_grain,
                    T_melt=T_melt,
                )
                Nu = IceConvection.nusselt_number(Ra)
                is_convecting = Ra >= ConvectionConstants.RA_CRIT
                idx_c = np.searchsorted(z_grid, z_c)

            Ti_fallback = (T_melt + T_c) / 2
            state = ConvectionState(
                idx_c=idx_c, z_c=z_c, D_cond=D_cond, D_conv=D_conv,
                T_c=T_c, Ti=Ti_fallback, Ra=Ra, Nu=Nu, is_convecting=is_convecting,
                nu_scaling=nu_scaling,
            )
        elif nu_scaling in ("green", "isoviscous_benchmark"):
            # Phase 2: scan temperature profile
            state = IceConvection.scan_temperature_profile(
                T_profile=T_profile,
                z_grid=z_grid,
                total_thickness=total_thickness,
                T_melt=T_melt,
                Q_v=Q_v,
                Q_b=Q_b,
                d_grain=d_grain,
                p_grain=p_grain,
                use_composite_viscosity=use_composite_viscosity,
                eta_ref=eta_ref,
                use_composite_transition_closure=use_composite_transition_closure,
                d_del=d_del,
                D0v=D0v,
                D0b=D0b,
            )
        elif nu_scaling == "dv2021":
            # DV2021: scan temperature profile (same as Green pathway) — the
            # DV2021-specific Nu computation is handled inside scan_temperature_profile.
            state = IceConvection.scan_temperature_profile(
                T_profile=T_profile,
                z_grid=z_grid,
                total_thickness=total_thickness,
                T_melt=T_melt,
                Q_v=Q_v,
                Q_b=Q_b,
                d_grain=d_grain,
                p_grain=p_grain,
                use_composite_viscosity=use_composite_viscosity,
                eta_ref=eta_ref,
                use_composite_transition_closure=False,
                d_del=d_del,
                D0v=D0v,
                D0b=D0b,
            )
        else:
            raise ValueError(f"Unknown nu_scaling={nu_scaling!r}")

        if use_onset_consistent_partition:
            state = IceConvection._collapse_subcritical_state(state, total_thickness, N_nodes)

        # Allow external modification of convection state before k_profile is built
        if convection_adjuster is not None:
            convection_adjuster(state, T_profile, z_grid, total_thickness, q_ocean)

        # Base conductivity from temperature
        k_profile = Thermal.conductivity(T_profile).copy()
        
        # Apply porosity correction (cold porous layer)
        if porosity > 0:
            T_phi = porosity_cure_temp if porosity_cure_temp is not None else Thermal.POR_CUR_TEMP_MEAN
            is_porous = T_profile < T_phi
            k_profile = np.where(is_porous, k_profile * (1 - porosity), k_profile)
        
        # Apply salt scaling
        if salt_fraction > 0:
            k_profile = k_profile * (1 + salt_fraction * (salt_scaling_factor - 1.0))
        
        # Apply Nu enhancement in convective layer ONLY (below idx_c).
        # --- Pitfall #7 guard (DV2021): k_eff = Nu * k must be applied only in
        # the convecting sublayer (z >= z_c), NOT in the conductive lid above.
        # The slice k_profile[idx_c:] ensures the lid (indices 0..idx_c-1) is
        # unmodified.  Smooth ramp: Nu_eff = 1 + ramp * (Nu_raw - 1).
        if state.is_convecting and state.idx_c < N_nodes:
            Nu_eff = 1.0 + nu_ramp_factor * (state.Nu - 1.0)
            # Guard: lid conductivity must remain unenhanced
            k_lid_before = k_profile[:state.idx_c].copy() if state.idx_c > 0 else np.array([])
            k_profile[state.idx_c:] *= Nu_eff
            if state.idx_c > 0:
                assert np.allclose(k_profile[:state.idx_c], k_lid_before), (
                    "Nu enhancement leaked into conductive lid (indices < idx_c). "
                    "k_eff = Nu * k must only apply in the convecting sublayer."
                )
        
        return k_profile, state

    @staticmethod
    def harmonic_mean_vectorized(
            k_profile: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """
        Vectorized computation of half-node harmonic mean conductivities.
        
        Returns k_{i+1/2} for i = 0 to N-2 (N-1 values total).
        
        Args:
            k_profile: Nodal conductivity array (W/m·K)
            
        Returns:
            k_half: Array of half-node conductivities k_{i+1/2}
        """
        k_i = k_profile[:-1]
        k_ip1 = k_profile[1:]
        
        # Harmonic mean: 2*k_i*k_{i+1} / (k_i + k_{i+1})
        return 2.0 * k_i * k_ip1 / (k_i + k_ip1 + 1e-30)
