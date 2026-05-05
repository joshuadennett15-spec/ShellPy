"""
Europa Ice Shell Thermal Solver

Implementation of a 1D transient heat conduction equation:
- Crank-Nicolson time discretization (θ=0.5, 2nd order)
- Rannacher startup (Backward Euler at half-timestep)
- Flux-conservative differencing for variable conductivity
- Optional spherical coordinate geometry

Phase 2 Enhancements (Green et al. 2021, Deschamps & Vilella 2021):
- Dynamic conductive/convective interface detection via temperature profile scanning
- Harmonic mean averaging at conductivity interfaces for numerical stability
- Explicit D_cond and D_conv tracking at each timestep

This solver utilizes IcePhysics for material properties and SurfaceCondition for boundary conditions,

References:
    - Rannacher, R. (1984): Finite element solution of diffusion problems
    - Howell (2021): Europa ice shell thermal modeling
    - Green et al. (2021): Parameterized convection methodology
    - Deschamps & Vilella (2021): Nu-Ra scaling laws
"""

import numpy as np
import numpy.typing as npt
from typing import Optional, Dict, Any, Union
from scipy.linalg import solve_banded

#Import previous modules
from constants import Thermal, Planetary, Rheology, HeatFlux, FloatOrArray
from Physics import IcePhysics
from Convection import IceConvection, ConvectionState
from Boundary_Conditions import surfacecondition, FixedTemperature, check_equilibrium

class Thermal_Solver:
    """
    Finite-Difference solver for 1D transient heat conduction in ice shells.

    Example:
        Solver = Thermal_Solver(nx=101, thickness=10e3)
        solver.run_to_equilibrium(q_ocean=0.01)
    """

    def __init__(self, nx: int = 101, thickness: float = 10e3, dt: float = 1e11, total_time: float = 1e15,
                 coordinate_system: str = 'auto', surface_bc: Optional[surfacecondition] = None,
                 rannacher_steps: int = 4, use_convection: bool = False,
                 physics_params: Optional[Dict[str, float]] = None,
                 convection_adjuster=None):
        """
        Initialize the thermal solver.

        Args:
            nx: Number of spatial nodes
            thickness: Initial ice shell thickness (m)
            dt: Time step (s), default ~3170 years
            total_time: Total simulation time (s)
            coordinate_system: 'cartesian', 'spherical', or 'auto'
            surface_bc: Surface boundary condition (SurfaceCondition instance)
            rannacher_steps: Number of BE startup steps (0 to disable)
            use_convection: Enable stagnant-lid convection parameterization
            physics_params: Optional dict of sampled physics parameters
                            (d_grain, Q_v, Q_b, epsilon_0, mu_ice, etc.)
        """
        # Grid parameters
        self.nx: int = nx
        self.H: float = thickness
        self.dt: float = dt
        self.total_time: float = total_time
        self.total_steps = int(total_time / dt)

        # time stepping
        self.rannacher_steps: int = rannacher_steps
        self.current_step: int = 0

        # Convection
        self.use_convection = use_convection
        self._convection_adjuster = convection_adjuster
        self._current_q_ocean = 0.0

        # Phase 2: Convection state tracking (Green et al. 2021)
        self.convection_state: Optional[ConvectionState] = None

        # Sampled physics parameters (for Monte Carlo)
        self.phys = physics_params or {}
        self.use_composite_transition_closure = bool(
            self.phys.get('use_composite_transition_closure', False)
        )
        self.use_onset_consistent_partition = bool(
            self.phys.get('use_onset_consistent_partition', False)
        )

        # Convection ramp factor: 0 = pure conduction, 1 = full convection
        # Used to smooth the transition for cold polar columns
        self.convection_ramp: float = 1.0

        # Cache for CN explicit-side material properties (reused across Picard iterations)
        self._cn_explicit_cache: Optional[Dict[str, Any]] = None

        # Coordinate system (select spherical for thick shells)
        if coordinate_system == 'auto':
            self.coordinate_system = 'spherical' if thickness >= 30e3 else 'cartesian'
        else:
            self.coordinate_system = coordinate_system

        # Normalised grid (0 = surface, 1 = base)
        self.nodes = np.linspace(0, 1, nx)

        # Boundary Conditions (set before initial profile so we can query T_surf)
        self.surface_bc = surface_bc or FixedTemperature(Thermal.SURFACE_TEMP_MEAN)

        # Initial temperature profile (linear geotherm)
        # Use actual surface BC temperature and pressure-dependent basal melting
        T_surface = self.surface_bc.get_temperature(time=0.0, current_T=Thermal.SURFACE_TEMP_MEAN,
                                                     k=3.0, dz=thickness / (nx - 1))
        T_base = float(IcePhysics.basal_melting_point(thickness))
        self.T = np.linspace(T_surface, T_base, nx)

        # allocate work arrays
        self._diag_main = np.ones(nx)
        self._diag_lower = np.zeros(nx - 1)
        self._diag_upper = np.zeros(nx - 1)
        self._rhs = np.zeros(nx)

    # =========================================================================
    # GRID UTILITIES
    # =========================================================================

    def _get_depths(self) -> npt.NDArray[np.float64]:
        """Returns physical depths from surface (m)."""
        return self.nodes * self.H

    def _get_radii(self) -> npt.NDArray[np.float64]:
        """Returns physical radii from surface (m)."""
        return Planetary.RADIUS - self._get_depths()

    def _get_dz(self) -> float:
        """returns current grid spacing (m)."""
        return self.H / (self.nx - 1)

    def _get_tidal_heating(self, T_array: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Calculates volumetric tidal heating (W/m^3).
        
        Uses IcePhysics.tidal_heating with sampled parameters.
        When use_convection is enabled, uses composite viscosity for
        grain-size dependent heating.
        """
        eps = self.phys.get('epsilon_0', 1e-5)
        d_grain = self.phys.get('d_grain', 1e-3)
        mu_ice = self.phys.get('mu_ice')
        Q_diff = self.phys.get('Q_v')  # Activation energy
        Q_gbs = self.phys.get('Q_b')
        D0v = self.phys.get('D0v')
        D0b = self.phys.get('D0b')
        d_del = self.phys.get('d_del')
        p_grain = self.phys.get('p_grain')

        # Use composite viscosity if convection is enabled
        q_vol = IcePhysics.tidal_heating(
            T_array,
            epsilon_0=eps,
            mu_ice=mu_ice,
            use_composite_viscosity=self.use_convection,
            d_grain=d_grain,
            Q_diff=Q_diff,
            Q_gbs=Q_gbs,
            D0v=D0v,
            D0b=D0b,
            d_del=d_del,
            p_grain=p_grain,
        )
        
        return q_vol

    def _effective_nusselt_number(self, raw_nu: float) -> float:
        """Returns the transport-effective Nusselt number after ramping."""
        return 1.0 + self.convection_ramp * (raw_nu - 1.0)

    def _compute_material_properties(
            self,
            T_profile: npt.NDArray[np.float64],
            *,
            update_convection_state: bool = False,
    ) -> Dict[str, Any]:
        """
        Compute the exact solver-side material state for a temperature profile.

        This centralizes conductivity-profile construction so diagnostics and
        matrix assembly use the same convection partitioning and Nu ramping.
        """
        depths = self._get_depths()

        if self.use_convection:
            k, convection_state = IceConvection.build_conductivity_profile(
                T_profile=T_profile,
                z_grid=depths,
                total_thickness=self.H,
                T_melt=IcePhysics.basal_melting_point(self.H),
                Q_v=self.phys.get('Q_v'),
                Q_b=self.phys.get('Q_b'),
                d_grain=self.phys.get('d_grain'),
                p_grain=self.phys.get('p_grain'),
                use_composite_viscosity=True,
                use_composite_transition_closure=self.use_composite_transition_closure,
                d_del=self.phys.get('d_del'),
                D0v=self.phys.get('D0v'),
                D0b=self.phys.get('D0b'),
                porosity=self.phys.get('f_porosity', 0.0),
                salt_fraction=self.phys.get('f_salt', 0.0),
                salt_scaling_factor=self.phys.get('B_k', 1.0),
                porosity_cure_temp=self.phys.get('T_phi', Thermal.POR_CUR_TEMP_MEAN),
                nu_ramp_factor=self.convection_ramp,
                use_onset_consistent_partition=self.use_onset_consistent_partition,
                convection_adjuster=self._convection_adjuster,
                q_ocean=self._current_q_ocean,
            )
        else:
            k = IcePhysics.effective_conductivity(
                T_profile,
                porosity=self.phys.get('f_porosity', 0.0),
                salt_fraction=self.phys.get('f_salt', 0.0),
                salt_scaling_factor=self.phys.get('B_k', 1.0),
                porosity_cure_temp=self.phys.get('T_phi', Thermal.POR_CUR_TEMP_MEAN),
            )
            convection_state = None

        if update_convection_state:
            self.convection_state = convection_state

        return {
            'k': k,
            'rho': Thermal.density_ice(T_profile),
            'cp': Thermal.specific_heat(T_profile),
            'q_tidal': self._get_tidal_heating(T_profile),
            'convection_state': convection_state,
        }

    def get_profile_diagnostics(self) -> Dict[str, Any]:
        """Return exact solver-side profiles for the current equilibrium state."""
        props = self._compute_material_properties(
            self.T,
            update_convection_state=True,
        )
        return {
            'temperature_K': self.T.copy(),
            'depth_m': self._get_depths().copy(),
            'tidal_heating_W_m3': props['q_tidal'].copy(),
            'effective_conductivity_W_mK': props['k'].copy(),
            'density_kg_m3': props['rho'].copy(),
            'specific_heat_J_kgK': props['cp'].copy(),
            'convection': self.get_convection_diagnostics(),
        }

    # =========================================================================
    # TIME STEPPING CONTROL
    # =========================================================================

    def _get_theta_and_dt(self) -> tuple[float, float]:
        """
        Returns time stepping parameters based on Rannacher schedule.

        Rannacher stepping:
            - Steps 0 to rannacher_steps-1: θ=1.0 (BE), dt/2
            - Steps >= rannacher_steps: θ=0.5 (CN), full dt

        Returns:
            (theta, effective_dt)
        """
        if self.current_step < self.rannacher_steps:
            return 1.0, self.dt / 2                 # Backward Euler, half step
        else:
            return 0.5, self.dt                     # Crank-Nicolson, full step

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # MATRIX ASSEMBLY
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    def _assemble_system(self, T_Coeff: npt.NDArray[np.float64],
                         T_explicit: npt.NDArray[np.float64]
                         )-> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
        """
        Assembles tridiagonal system using separate temperature profiles.
        
        Phase 2 Enhancement (Green et al. 2021, Deschamps & Vilella 2021):
        - Scans temperature profile to dynamically locate conductive/convective interface
        - Uses HARMONIC MEAN for half-node conductivities (flux-conservative)
        - Stores convection state for diagnostics (D_cond, D_conv, Ra, Nu)

        :param T_Coeff: Temperature for calculating material properties (current guess)
        :param T_explicit: Temperature at time n (fixed history for RHS)
        :return:
            (diag_lower, diag_main, diag_upper, b): Sparse diagonals and RHS
        """
        dz = self._get_dz()
        theta, dt_eff = self._get_theta_and_dt()
        n = self.nx
        depths = self._get_depths()

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # MATERIAL PROPERTIES (Based on T_coeff linearization)
        # Phase 2: Temperature-profile based convection parameterization
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        props = self._compute_material_properties(
            T_Coeff,
            update_convection_state=True,
        )
        k = props['k']
        rho = props['rho']
        cp = props['cp']
        q_tidal = props['q_tidal']

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # GEOMETRIC FACTORS (Spherical or Cartesian)
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        if self.coordinate_system == 'spherical':
            radii = self._get_radii()
            G = radii ** 2
        else:
            G = np.ones(n)

        # Half-node geometric factors
        G_plus = (G[:-1] + G[1:]) / 2
        G_minus = (G[1:] + G[:-1]) / 2

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # FLUX-CONSERVATIVE DIFFERENCING: Harmonic Mean Conductivity
        # (Green et al. 2021, Deschamps & Vilella 2021)
        # 
        # When k jumps by orders of magnitude at the interface (k -> Nu*k),
        # standard arithmetic averaging causes numerical instability.
        # Harmonic mean ensures flux conservation: k_{i+1/2} = 2*k_i*k_{i+1}/(k_i+k_{i+1})
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        if self.use_convection:
            # Use harmonic mean for flux-conservative differencing
            k_half = IceConvection.harmonic_mean_vectorized(k)
            k_plus = k_half  # k_{i+1/2} for i = 0 to n-2
            k_minus = k_half  # k_{i-1/2} = k_{i+1/2} shifted (same interfaces)
        else:
            # Standard arithmetic mean for non-convective case
            k_plus = (k[:-1] + k[1:]) / 2
            k_minus = (k[1:] + k[:-1]) / 2

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Vectorized Diffusion Factors
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        interior = slice(1, n - 1)
        G_interior = G[interior]

        factor_plus = k_plus[1:] * G_plus[1:] / (G_interior * dz ** 2)
        factor_minus = k_minus[:-1] * G_minus[:-1] / (G_interior * dz ** 2)

        # diffusion coefficient uses T_coeff properties
        alpha = dt_eff / (rho[interior] * cp[interior])

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Fill Diags
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        # reset identity for boundary rows
        self._diag_main[:] = 1.0
        self._diag_main[interior] = 1 + theta * alpha * (factor_plus + factor_minus)

        self._diag_lower[:] = 0.0
        self._diag_lower[:n-2] = -theta * alpha * factor_minus

        self._diag_upper[:] = 0.0
        self._diag_upper[1:n-1] = -theta * alpha * factor_plus

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Fill RHS Vector
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        self._rhs[:] = 0

        # T_explicit used for the explicit flux term
        T_diff_plus = T_explicit[2:] - T_explicit[1:-1]
        T_diff_minus = T_explicit[1:-1] - T_explicit[:-2]

        if theta < 1.0:
            # Crank-Nicolson explicit flux: requires properties evaluated at T^n.
            # Cache material properties since T_explicit is constant across Picard iterations.
            if self._cn_explicit_cache is None:
                explicit_props = self._compute_material_properties(
                    T_explicit,
                    update_convection_state=False,
                )
                k_exp = explicit_props['k']
                if self.use_convection:
                    k_half_exp = IceConvection.harmonic_mean_vectorized(k_exp)
                    cached_k_plus = k_half_exp
                    cached_k_minus = k_half_exp
                else:
                    cached_k_plus = (k_exp[:-1] + k_exp[1:]) / 2
                    cached_k_minus = (k_exp[1:] + k_exp[:-1]) / 2

                self._cn_explicit_cache = {
                    'k_plus': cached_k_plus,
                    'k_minus': cached_k_minus,
                    'rho': explicit_props['rho'],
                    'cp': explicit_props['cp'],
                }

            cache = self._cn_explicit_cache
            alpha_exp = dt_eff / (cache['rho'][interior] * cache['cp'][interior])

            factor_plus_exp = cache['k_plus'][1:] * G_plus[1:] / (G_interior * dz ** 2)
            factor_minus_exp = cache['k_minus'][:-1] * G_minus[:-1] / (G_interior * dz ** 2)

            explicit_coeff = (1 - theta) * alpha_exp
            flux = factor_plus_exp * T_diff_plus - factor_minus_exp * T_diff_minus
        else:
            flux = 0.0
            explicit_coeff = 0.0

        # Source term is evaluated at T_Coeff (implicit/guess)
        source = (dt_eff * q_tidal[interior]) / (rho[interior] * cp[interior])

        self._rhs[interior] = T_explicit[interior] + explicit_coeff * flux + source

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Boundary conditions
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        a, b_coef, c = self.surface_bc.get_linearization(
            time=self.current_step * self.dt,
            current_T=T_Coeff[0],
            k=k[0],
            dz=dz
        )
        self._diag_main[0] = a
        self._diag_upper[0] = -c
        self._rhs[0] = -b_coef

        self._rhs[-1] = IcePhysics.basal_melting_point(self.H)

        return self._diag_lower, self._diag_main, self._diag_upper, self._rhs

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Boundary conditions
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    def _solve_tridiagonal(self, diag_lower, diag_main, diag_upper, rhs):
        """Solve tridiagonal system Ax = b using scipy.linalg.solve_banded."""
        n = len(rhs)
        # Pack into banded form: row 0 = upper diagonal, row 1 = main, row 2 = lower
        ab = np.zeros((3, n))
        ab[0, 1:] = diag_upper       # upper diagonal (offset by 1)
        ab[1, :] = diag_main         # main diagonal
        ab[2, :-1] = diag_lower      # lower diagonal
        return solve_banded((1, 1), ab, rhs)

    def solve_step(self, q_ocean: float) -> float:
        """
        Advances the solution by one time step.

        Args:
            q_ocean: Heat flux from ocean (W/m²)

        Returns:
            db_dt: Freezing front velocity (m/s)
        """
        _, dt_eff = self._get_theta_and_dt()

        self._current_q_ocean = q_ocean

        # Store T^n for the explicit part
        T_old = self.T.copy()
        T_guess = T_old.copy()

        # Clear CN explicit-side cache (T_explicit is new this step)
        self._cn_explicit_cache = None

        # --- Pitfall #5 verified: Predictor-corrector Picard iteration exists.
        # 3 iterations with 0.01 K convergence tolerance.  Each iteration
        # re-evaluates temperature-dependent coefficients (k(T), eta(T)) at the
        # latest guess, then solves the tridiagonal system.  This prevents the
        # single-pass lagging error that can reach ~5% in k(T) per timestep.
        for _ in range(3):
            diag_lower, diag_main, diag_upper, b = self._assemble_system(
                T_Coeff=T_guess, T_explicit=T_old
            )

            T_new = self._solve_tridiagonal(diag_lower, diag_main, diag_upper, b)

            if np.max(np.abs(T_new - T_guess)) < 0.01:
                T_guess = T_new
                break

            T_guess = T_new

        # Update state
        self.T = T_guess

        # Stefan condition: update shell thickness
        dz = self._get_dz()
        k_basal = IcePhysics.effective_conductivity(self.T[-1])
        db_dt = IcePhysics.stefan_velocity(self.T, dz, q_ocean, k_basal)

        self.H += db_dt * dt_eff
        self.H = max(self.H, 500.0)  # Floor: 500 m minimum thickness

        # Rannacher: second half-step during startup phase
        # NOTE: current_step is incremented AFTER both half-steps so that
        # _get_theta_and_dt() returns BE/dt_half for both halves.
        if self.current_step < self.rannacher_steps:
            T_old_2 = self.T.copy()
            T_guess_2 = T_old_2.copy()

            # Clear cache for second half-step (new T_explicit)
            self._cn_explicit_cache = None

            for _ in range(3):
                dl, dm, du, b2 = self._assemble_system(T_guess_2, T_old_2)
                T_new_2 = self._solve_tridiagonal(dl, dm, du, b2)

                if np.max(np.abs(T_new_2 - T_guess_2)) < 0.01:
                    T_guess_2 = T_new_2
                    break
                T_guess_2 = T_new_2

            self.T = T_guess_2

            # Update thickness for the second half-step too
            dz = self._get_dz()
            k_basal2 = IcePhysics.effective_conductivity(self.T[-1])
            db_dt2 = IcePhysics.stefan_velocity(self.T, dz, q_ocean, k_basal2)
            self.H += db_dt2 * dt_eff
            self.H = max(self.H, 500.0)  # Floor: 500 m minimum thickness
            db_dt = (db_dt + db_dt2) / 2

        # Increment step AFTER both half-steps complete
        self.current_step += 1

        return db_dt

    def run_to_equilibrium(
        self, q_ocean: float,
        threshold: float = 1e-15,
        max_steps: Optional[int] = None,
        log_interval: int = 100
    ) -> Dict[str, Any]:
        """
        Runs until thermal equilibrium (db/dt → 0).
        Args:
            q_ocean: Ocean heat flux (W/m²)
            threshold: Velocity threshold for equilibrium (m/s)
            max_steps: Maximum steps (None = use total_steps)
            log_interval: Steps between progress logs

        Returns:
            Dictionary with final state and history
        """
        max_steps = max_steps or self.total_steps

        thickness_history = [self.H / 1000]
        velocity_history = []

        for step in range(max_steps):
            velocity = self.solve_step(q_ocean)

            thickness_history.append(self.H / 1000)
            velocity_history.append(velocity)

            if check_equilibrium(velocity, threshold):#
                print(f"\n[OK] Equilibrium at step {step}")
                print(f"  Thickness: {self.H/1000:.2f} km ")
                print (f"  Velocity: {velocity:.2e} m/s")
                break

            if step % log_interval == 0:
                phase = "BE" if step < self.rannacher_steps else "CN"
                print(f"Step {step:5d} [{phase}]: H = {self.H / 1000:6.2f} km, db/dt = {velocity:+.2e} m/s")

        return {
            'final_thickness_km': self.H / 1000,
            'final_temperature': self.T.copy(),
            'thickness_history': np.array(thickness_history),
            'velocity_history': np.array(velocity_history),
            'steps': step + 1,
            'coordinate_system': self.coordinate_system,
        }

    def get_diagnostics(self) -> Dict[str, Any]:
        """Returns current state diagnostics."""
        diagnostics = {
            'thickness_km': self.H / 1000,
            'T_surface': self.T[0],
            'T_base': self.T[-1],
            'coordinate_system': self.coordinate_system,
            'current_step': self.current_step,
            'in_rannacher': self.current_step < self.rannacher_steps,
        }
        
        # Phase 2: Add convection state diagnostics
        if self.convection_state is not None:
            diagnostics.update(self.get_convection_diagnostics())
        
        return diagnostics

    def get_convection_diagnostics(self) -> Dict[str, Any]:
        """
        Returns Phase 2 convection-specific diagnostics.
        
        Includes D_cond, D_conv, Ra, Nu as required by the parameterized
        convection methodology (Green et al. 2021).
        
        Returns:
            Dictionary with convection state parameters, or empty dict if
            convection is disabled or no state available.
        """
        if self.convection_state is None:
            return {}
        
        state = self.convection_state
        nu_eff = self._effective_nusselt_number(state.Nu)
        return {
            'D_cond_km': state.D_cond / 1000.0,
            'D_conv_km': state.D_conv / 1000.0,
            'z_c_km': state.z_c / 1000.0,
            'idx_c': state.idx_c,
            'T_c': state.T_c,
            'Ti': state.Ti,
            'Ra': state.Ra,
            'Nu': nu_eff,
            'Nu_raw': state.Nu,
            'convection_ramp': self.convection_ramp,
            'is_convecting': state.is_convecting,
            'lid_fraction': state.D_cond / self.H if self.H > 0 else 0.0,
        }
