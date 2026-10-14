"""Time-marching solver for one ice-shell column with a freezing/melting base.

Heat equation rho cp dT/dt = div(k grad T) + q_tidal in a spherical shell, on a grid that
stretches with the shell thickness H (0 = surface, 1 = base). Surface: fixed temperature. Base:
melting point, and the Stefan condition moves H. Backward Euler in time (only the equilibrium
matters, and unlike Crank-Nicolson it damps grid-scale oscillations at these long steps),
flux-conservative finite differences in depth, and Picard iteration on the
temperature-dependent coefficients.
"""
import numpy as np
from scipy.linalg import solve_banded

from shellpy import constants as c
from shellpy import convection, physics


class ThermalSolver:
    """One column. p holds the sampled shell parameters; H0 is the starting thickness (m)."""

    def __init__(self, p, H0, *, nx=121, dt=1e12):
        self.p = p
        self.H = H0
        self.nx = nx
        self.dt = dt
        self.steps = 0
        self.converged = False
        self.xi = np.linspace(0, 1, nx)                       # normalised depth
        self.T = np.linspace(p['T_surf'], physics.melting_point(H0), nx)
        self.state = None                                     # ConvectionState of the latest solve

    def properties(self, T):
        """Conductivity (with convection), density, heat capacity, tidal heating and convection state."""
        p = self.p
        k, state = convection.conductivity_profile(T, self.xi * self.H, self.H, physics.melting_point(self.H), p)
        eta = physics.diffusion_viscosity(T, p)
        q_tidal = physics.andrade_heating(eta, p['epsilon_0'], p['mu_ice'])
        return k, physics.density(T), physics.specific_heat(T), q_tidal, state

    def run(self, q_ocean, max_steps=50_000, tol=1e-15, window=200):
        """Step until |dH/dt| < tol (m/s) for `window` consecutive steps; returns self.converged."""
        calm = 0
        while self.steps < max_steps and calm < window:
            calm = calm + 1 if abs(self.step(q_ocean)) < tol else 0
        self.converged = calm >= window
        return self.converged

    def step(self, q_ocean):
        """Advance one time step; returns the freezing-front velocity (m/s)."""
        T_old = self.T
        T = T_old
        for _ in range(3):   # Picard iterations on the temperature-dependent coefficients
            T_new, q_ice = self._solve_linear(T, T_old)
            converged = np.max(np.abs(T_new - T)) < 0.01
            T = T_new
            if converged:
                break
        if not np.all(np.isfinite(T)):
            raise FloatingPointError(f"non-finite temperature at step {self.steps}, H = {self.H:.0f} m")
        self.T = T
        v = physics.stefan_velocity(q_ice, q_ocean, T[-1])
        self.H = max(self.H + v * self.dt, 500.0)
        self.steps += 1
        return v

    def _face_factors(self, k, state):
        """Conductances for interior nodes to their lower (fp) and upper (fm) neighbours, and the
        bottom face's conductivity scaled to the base area."""
        dz2 = (self.H / (self.nx - 1)) ** 2
        G = (c.RADIUS - self.xi * self.H) ** 2   # area factor r^2 of a spherical shell
        kG = convection.face_conductivity(k, state, self.xi * self.H) * ((G[:-1] + G[1:]) / 2)
        return kG[1:] / (G[1:-1] * dz2), kG[:-1] / (G[1:-1] * dz2), kG[-1] / G[-1]

    def _solve_linear(self, T_coeff, T_old):
        """One backward-Euler solve with coefficients evaluated at T_coeff.

        Returns the new temperatures and the heat flux conducted into the ice at its base:
        the flux through the bottom cell face (Nu-enhanced where convecting) minus the tidal
        heat generated in the bottom half cell, so the shell's energy budget closes.
        """
        k, rho, cp, q_tidal, self.state = self.properties(T_coeff)
        fp, fm, k_base = self._face_factors(k, self.state)
        a = self.dt / (rho[1:-1] * cp[1:-1])
        ab = np.zeros((3, self.nx))                           # banded matrix: upper, main, lower diagonals
        ab[0, 2:] = -a * fp
        ab[1] = 1.0
        ab[1, 1:-1] = 1 + a * (fp + fm)
        ab[2, :-2] = -a * fm
        rhs = np.empty(self.nx)
        rhs[0] = self.p['T_surf']
        rhs[1:-1] = T_old[1:-1] + a * q_tidal[1:-1]
        rhs[-1] = physics.melting_point(self.H)
        T = solve_banded((1, 1), ab, rhs)
        dz = self.H / (self.nx - 1)
        return T, k_base * (T[-1] - T[-2]) / dz - q_tidal[-1] * dz / 2
