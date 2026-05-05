"""
2D Axisymmetric Coupled-Column Solver for Europa's Ice Shell.

Uses operator splitting: radial heat transport (implicit, via Thermal_Solver)
coupled with lateral heat diffusion (explicit) across latitude columns.

Convention: phi = 0 at equator, phi = pi/2 at pole (geographic latitude).

References:
    - Design spec: Europa2D/docs/2026-03-12-2d-axisymmetric-model-design.md
    - Howell (2021), Green et al. (2021), Deschamps & Vilella (2021)
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'Europa1D', 'src'))

import numpy as np
import numpy.typing as npt
from typing import Dict, Literal, Optional, Any, List
from scipy.linalg import solve_banded

from Solver import Thermal_Solver
from Boundary_Conditions import FixedTemperature
from Physics import IcePhysics
from constants import Planetary, Thermal

from latitude_profile import LatitudeProfile
from convection_2d import ConvectionHypothesis, make_adjuster


class AxialSolver2D:
    """
    Coupled-column 2D axisymmetric thermal solver.

    Maintains N latitude columns, each a 1D Thermal_Solver.
    Columns are coupled by explicit lateral heat diffusion.
    """

    def __init__(
        self,
        n_lat: int = 19,
        nx: int = 31,
        dt: float = 1e12,
        total_time: float = 5e14,
        latitude_profile: Optional[LatitudeProfile] = None,
        physics_params: Optional[Dict[str, float]] = None,
        use_convection: bool = True,
        initial_thickness: float = 20e3,
        rannacher_steps: int = 4,
        coordinate_system: str = 'auto',
        lateral_method: Literal['implicit', 'explicit'] = 'implicit',
        hypothesis: Optional[ConvectionHypothesis] = None,
    ):
        """
        Initialize the 2D axisymmetric solver.

        Args:
            n_lat: Number of latitude columns (equator to pole)
            nx: Radial nodes per column
            dt: Time step (s)
            total_time: Total simulation time (s)
            latitude_profile: Latitude-dependent parameter functions
            physics_params: Shared MC-sampled parameters (grain size, etc.)
            use_convection: Enable stagnant-lid convection
            initial_thickness: Starting ice shell thickness for all columns (m)
            rannacher_steps: Number of Backward Euler startup steps
            coordinate_system: 'auto', 'cartesian', or 'spherical'
            lateral_method: 'implicit' (tridiagonal solve, unconditionally stable)
                or 'explicit' (forward Euler, retained for validation)
        """
        self.n_lat = n_lat
        self.nx = nx
        self.dt = dt
        self.total_time = total_time
        self.use_convection = use_convection
        self.lateral_method = lateral_method
        self.hypothesis = hypothesis
        self.profile = latitude_profile or LatitudeProfile()
        self._shared_params = dict(physics_params or {})
        if use_convection:
            # Opt into the richer transition closure for 2D columns without
            # changing legacy 1D solver defaults.
            self._shared_params.setdefault('use_composite_transition_closure', True)
            self._shared_params.setdefault('use_onset_consistent_partition', True)

        # Geographic latitude grid: 0 (equator) to pi/2 (pole)
        self.latitudes = np.linspace(0, np.pi / 2, n_lat)
        self.dphi = self.latitudes[1] - self.latitudes[0] if n_lat > 1 else 1.0

        # Build one 1D solver per latitude column.
        # Each column uses the shared 1D convection logic directly.

        self.columns: List[Thermal_Solver] = []
        for j in range(n_lat):
            phi_j = self.latitudes[j]

            # Per-column overrides from latitude profile
            col_params = dict(self._shared_params)
            lat_vals = self.profile.evaluate_at(phi_j)
            col_params['epsilon_0'] = lat_vals['epsilon_0']
            col_params['T_surf'] = lat_vals['T_surf']

            # Latitude-dependent grain size: dynamic recrystallization
            # Higher tidal strain at poles → smaller grains
            grain_scale = lat_vals.get('grain_scale', 1.0)
            if 'd_grain' in col_params and grain_scale != 1.0:
                col_params['d_grain'] = col_params['d_grain'] * grain_scale

            surface_bc = FixedTemperature(temperature=lat_vals['T_surf'])

            # Create per-column convection adjuster from hypothesis
            adjuster = make_adjuster(self.hypothesis, phi_j, self.profile)

            solver = Thermal_Solver(
                nx=nx,
                thickness=initial_thickness,
                dt=dt,
                total_time=total_time,
                coordinate_system=coordinate_system,
                surface_bc=surface_bc,
                rannacher_steps=rannacher_steps,
                use_convection=use_convection,
                physics_params=col_params,
                convection_adjuster=adjuster,
            )

            self.columns.append(solver)

    def get_thickness_profile(self) -> npt.NDArray[np.float64]:
        """Returns ice shell thickness at each latitude (m). Columns are authoritative."""
        return np.array([col.H for col in self.columns])

    def get_latitudes_deg(self) -> npt.NDArray[np.float64]:
        """Returns latitude grid in degrees."""
        return np.degrees(self.latitudes)

    def _lateral_diffusion_step(self) -> None:
        """
        Apply explicit lateral heat diffusion between columns.

        Uses the geographic latitude form of the diffusion operator:
            dT/dt = (k / (R^2 cos(phi))) * d/dphi[cos(phi) * dT/dphi]

        Boundary conditions: dT/dphi = 0 at equator (phi=0) and pole (phi=pi/2).

        Grid mismatch note: columns with different H map the same node index
        to different physical depths. Since lateral diffusion is extremely weak
        (tau ~ 200 Gyr), this introduces only second-order errors in an already
        negligible correction.
        """
        if self.n_lat < 3:
            return

        R = Planetary.RADIUS
        dphi = self.dphi
        _, dt_eff = self.columns[0]._get_theta_and_dt()

        # Build 2D temperature array: shape (n_lat, nx)
        T_2d = np.array([col.T for col in self.columns])

        # Metric factors at node points: cos(phi_j)
        cos_phi = np.cos(self.latitudes)

        # Half-node metric factors: cos(phi_{j+1/2})
        phi_half = (self.latitudes[:-1] + self.latitudes[1:]) / 2
        cos_half = np.cos(phi_half)

        # Mean thermal properties per column
        k_cols = np.array([
            float(Thermal.conductivity(np.mean(col.T))) for col in self.columns
        ])
        rho_cp_cols = np.array([
            float(Thermal.density_ice(np.mean(col.T))) * Thermal.SPECIFIC_HEAT
            for col in self.columns
        ])

        # Compute dT for all columns
        dT = np.zeros_like(T_2d)

        # Interior columns (j = 1 to n_lat-2)
        for j in range(1, self.n_lat - 1):
            alpha = k_cols[j] * dt_eff / (rho_cp_cols[j] * R**2 * dphi**2)
            flux_plus = cos_half[j] * (T_2d[j + 1, :] - T_2d[j, :])
            flux_minus = cos_half[j - 1] * (T_2d[j, :] - T_2d[j - 1, :])
            dT[j, :] = alpha * (flux_plus - flux_minus) / cos_phi[j]

        # Boundary: equator (phi=0), dT/dphi = 0 => flux_minus = 0
        j = 0
        alpha = k_cols[0] * dt_eff / (rho_cp_cols[0] * R**2 * dphi**2)
        flux_plus = cos_half[0] * (T_2d[1, :] - T_2d[0, :])
        dT[0, :] = alpha * flux_plus / max(cos_phi[0], 1e-10)

        # Boundary: pole (phi=pi/2), symmetry dT/dphi = 0.
        # The geographic-latitude diffusion operator:
        #   L[T] = (1/(R^2 cos(phi))) d/dphi[cos(phi) dT/dphi]
        # is singular at phi=pi/2 because cos(pi/2) = 0.
        #
        # L'Hopital's rule gives the well-defined limit:
        #   L[T] -> 2 d^2T/dphi^2 / R^2
        #
        # With the symmetry ghost node (T[j+1] = T[j-1]):
        #   d^2T/dphi^2 ≈ 2(T[j-1] - T[j]) / dphi^2
        #
        # Combined: L[T] = 4(T[j-1] - T[j]) / (R^2 dphi^2)
        j = self.n_lat - 1
        alpha = k_cols[j] * dt_eff / (rho_cp_cols[j] * R**2 * dphi**2)
        dT[j, :] = 4.0 * alpha * (T_2d[j - 1, :] - T_2d[j, :])

        # Writeback: modify each column's T directly
        for j in range(self.n_lat):
            self.columns[j].T += dT[j, :]

    def _lateral_diffusion_step_implicit(self) -> None:
        """
        Implicit lateral heat diffusion between columns.

        Solves  (I - dt·L_lat) T_new = T_old  per radial level,
        where L_lat is the geographic-latitude diffusion operator.
        Unconditionally stable — no CFL constraint on the lateral step.

        Same physics as the explicit method: L'Hopital correction at the
        pole, Neumann (dT/dphi = 0) at equator and pole.  The only
        difference is time integration: backward Euler instead of
        forward Euler.
        """
        if self.n_lat < 3:
            return

        R = Planetary.RADIUS
        dphi = self.dphi
        _, dt_eff = self.columns[0]._get_theta_and_dt()

        # Build 2D temperature array: shape (n_lat, nx)
        T_2d = np.array([col.T for col in self.columns])

        # Metric factors
        cos_phi = np.cos(self.latitudes)
        phi_half = (self.latitudes[:-1] + self.latitudes[1:]) / 2
        cos_half = np.cos(phi_half)

        # Per-column diffusion coefficient: alpha_j = k·dt / (rho·cp·R²·dphi²)
        alpha = np.array([
            float(Thermal.conductivity(np.mean(col.T))) * dt_eff
            / (float(Thermal.density_ice(np.mean(col.T))) * Thermal.SPECIFIC_HEAT
               * R**2 * dphi**2)
            for col in self.columns
        ])

        # Assemble tridiagonal coefficients for (I - dt·L) T_new = T_old
        n = self.n_lat
        lower = np.zeros(n - 1)
        main = np.ones(n)
        upper = np.zeros(n - 1)

        # Interior columns: j = 1 .. n-2
        for j in range(1, n - 1):
            c_minus = alpha[j] * cos_half[j - 1] / cos_phi[j]
            c_plus = alpha[j] * cos_half[j] / cos_phi[j]
            main[j] += c_minus + c_plus
            lower[j - 1] -= c_minus
            upper[j] -= c_plus

        # Equator (j=0): dT/dphi = 0 => no flux from below.
        # Guard cos_phi[0]=0 to match the explicit step (line 191) for grids
        # whose equator node lands exactly at phi=pi/2.
        c_plus_eq = alpha[0] * cos_half[0] / max(cos_phi[0], 1e-10)
        main[0] += c_plus_eq
        upper[0] -= c_plus_eq

        # Pole (j=n-1): L'Hopital limit => coefficient is 4·alpha
        main[n - 1] += 4.0 * alpha[n - 1]
        lower[n - 2] -= 4.0 * alpha[n - 1]

        # Pack into banded form for scipy.linalg.solve_banded
        ab = np.zeros((3, n))
        ab[0, 1:] = upper
        ab[1, :] = main
        ab[2, :-1] = lower

        # Solve per radial level (same matrix, different RHS)
        for i in range(self.nx):
            T_2d[:, i] = solve_banded((1, 1), ab, T_2d[:, i])

        # Writeback
        for j in range(n):
            self.columns[j].T = T_2d[j, :]

    def solve_step(self, q_ocean_profile: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Advance all columns by one time step using operator splitting.

        1. Radial step: each column solves its 1D heat equation independently
        2. Lateral step: diffusion between columns (implicit or explicit)

        Args:
            q_ocean_profile: Ocean heat flux at each latitude (W/m^2), shape (n_lat,)

        Returns:
            Array of freezing front velocities db/dt per column (m/s)
        """
        velocities = np.zeros(self.n_lat)

        # Step 1: Radial solve (independent per column)
        for j, col in enumerate(self.columns):
            velocities[j] = col.solve_step(q_ocean_profile[j])

        # Step 2: Lateral diffusion coupling
        if self.lateral_method == 'implicit':
            self._lateral_diffusion_step_implicit()
        else:
            self._lateral_diffusion_step()

        return velocities

    def run_to_equilibrium(
        self,
        threshold: float = 1e-12,
        max_steps: int = 1500,
        log_interval: int = 100,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Run all columns to thermal equilibrium.

        Convergence criterion: max(|db/dt|) across all columns < threshold.

        Args:
            threshold: Velocity threshold for equilibrium (m/s)
            max_steps: Maximum number of time steps
            log_interval: Steps between progress logs
            verbose: Print progress updates

        Returns:
            Dict with H_profile_km, T_2d, latitudes_deg, converged, steps, diagnostics
        """
        # Build ocean heat flux profile
        q_ocean_profile = np.array([
            self.profile.ocean_heat_flux(phi) for phi in self.latitudes
        ])

        converged = False
        final_step = 0

        for step in range(max_steps):
            velocities = self.solve_step(q_ocean_profile)

            max_vel = np.max(np.abs(velocities))
            final_step = step

            if max_vel < threshold:
                converged = True
                if verbose:
                    print(f"\n[OK] 2D equilibrium at step {step}")
                    H_km = self.get_thickness_profile() / 1000.0
                    print(f"  H range: {H_km.min():.2f} - {H_km.max():.2f} km")
                    print(f"  Max velocity: {max_vel:.2e} m/s")
                break

            if verbose and step % log_interval == 0:
                H_km = self.get_thickness_profile() / 1000.0
                print(f"Step {step:5d}: H = [{H_km.min():.2f}, {H_km.max():.2f}] km, "
                      f"max|v| = {max_vel:.2e} m/s")

        # Collect results
        H_km = self.get_thickness_profile() / 1000.0
        T_2d = np.array([col.T.copy() for col in self.columns])

        # Convection diagnostics per column
        diagnostics = []
        for col in self.columns:
            if col.convection_state is not None:
                diagnostics.append(col.get_convection_diagnostics())
            else:
                diagnostics.append({
                    'D_cond_km': col.H / 1000.0,
                    'D_conv_km': 0.0,
                    'T_c': 0.0,
                    'Ti': 0.0,
                    'z_c_km': col.H / 1000.0,
                    'Ra': 0.0,
                    'Nu': 1.0,
                    'Nu_raw': 1.0,
                    'convection_ramp': getattr(col, 'convection_ramp', 1.0),
                    'lid_fraction': 1.0,
                })

        return {
            'H_profile_km': H_km,
            'T_2d': T_2d,
            'latitudes_deg': self.get_latitudes_deg(),
            'converged': converged,
            'steps': final_step + 1,

            'diagnostics': diagnostics,
        }
