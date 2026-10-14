"""Time-marching solver for one ice-shell column with a freezing/melting base.

Heat equation rho cp dT/dt = div(k grad T) + q_tidal on a grid that stretches with
the shell thickness H (0 = surface, 1 = base). Surface: fixed temperature. Base:
melting point, and the Stefan condition moves H. Crank-Nicolson in time with a
backward-Euler start-up (Rannacher), flux-conservative finite differences in depth,
and Picard iteration on the temperature-dependent coefficients.
"""
import numpy as np
from scipy.linalg import solve_banded

from shellpy import constants as c
from shellpy import convection, physics


class ThermalSolver:
    """One column. p holds the sampled shell parameters; H0 is the starting thickness (m)."""

    def __init__(self, p, H0, *, nx=31, dt=1e12, spherical=None, rannacher_steps=4):
        self.p = p
        self.H = H0
        self.nx = nx
        self.dt = dt
        self.spherical = H0 >= 30e3 if spherical is None else spherical
        self.rannacher_steps = rannacher_steps
        self.steps = 0
        self.converged = False
        self.xi = np.linspace(0, 1, nx)                       # normalised depth
        self.T = np.linspace(p['T_surf'], physics.melting_point(H0), nx)
        self.state = None                                     # ConvectionState of the latest solve

    def properties(self, T):
        """Conductivity (with convection), density, heat capacity, tidal heating and convection state."""
        p = self.p
        k, state = convection.conductivity_profile(T, self.xi * self.H, self.H, physics.melting_point(self.H), p)
        eta = physics.diffusion_viscosity(T, p['d_grain'], p['d_del'], p['D0v'], p['D0b'], p['Q_v'], p['Q_b'])
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
        if self.steps < self.rannacher_steps:   # two backward-Euler half steps damp start-up oscillations
            v = (self._advance(q_ocean, 1.0, self.dt / 2) + self._advance(q_ocean, 1.0, self.dt / 2)) / 2
        else:
            v = self._advance(q_ocean, 0.5, self.dt)
        self.steps += 1
        return v

    def _advance(self, q_ocean, theta, dt):
        T_old = self.T
        explicit = self._explicit_flux(T_old, theta, dt) if theta < 1 else 0.0
        T = T_old
        for _ in range(3):   # Picard iterations on the temperature-dependent coefficients
            T_new, q_ice = self._solve_linear(T, T_old, theta, dt, explicit)
            converged = np.max(np.abs(T_new - T)) < 0.01
            T = T_new
            if converged:
                break
        if not np.all(np.isfinite(T)):
            raise FloatingPointError(f"non-finite temperature at step {self.steps}, H = {self.H:.0f} m")
        self.T = T
        v = physics.stefan_velocity(q_ice, q_ocean, T[-1])
        self.H = max(self.H + v * dt, 500.0)
        return v

    def _face_factors(self, k):
        """Conductances for interior nodes to their lower (fp) and upper (fm) neighbours, and the
        bottom face's conductivity scaled to the base area."""
        dz2 = (self.H / (self.nx - 1)) ** 2
        G = (c.RADIUS - self.xi * self.H) ** 2 if self.spherical else np.ones(self.nx)   # area factor r^2
        kG = convection.face_conductivity(k) * ((G[:-1] + G[1:]) / 2)
        return kG[1:] / (G[1:-1] * dz2), kG[:-1] / (G[1:-1] * dz2), kG[-1] / G[-1]

    def _explicit_flux(self, T_old, theta, dt):
        """Explicit (1 - theta) part of the Crank-Nicolson diffusion term, with coefficients at T_old."""
        k, rho, cp, _, _ = self.properties(T_old)
        fp, fm, _ = self._face_factors(k)
        a = dt / (rho[1:-1] * cp[1:-1])
        return (1 - theta) * a * (fp * (T_old[2:] - T_old[1:-1]) - fm * (T_old[1:-1] - T_old[:-2]))

    def _solve_linear(self, T_coeff, T_old, theta, dt, explicit):
        """One implicit solve with coefficients evaluated at T_coeff.

        Returns the new temperatures and the heat flux conducted into the ice at its base:
        the flux through the bottom cell face (Nu-enhanced where convecting) minus the tidal
        heat generated in the bottom half cell, so the shell's energy budget closes.
        """
        k, rho, cp, q_tidal, self.state = self.properties(T_coeff)
        fp, fm, k_base = self._face_factors(k)
        a = dt / (rho[1:-1] * cp[1:-1])
        ab = np.zeros((3, self.nx))                           # banded matrix: upper, main, lower diagonals
        ab[0, 2:] = -theta * a * fp
        ab[1] = 1.0
        ab[1, 1:-1] = 1 + theta * a * (fp + fm)
        ab[2, :-2] = -theta * a * fm
        rhs = np.empty(self.nx)
        rhs[0] = self.p['T_surf']
        rhs[1:-1] = T_old[1:-1] + explicit + dt * q_tidal[1:-1] / (rho[1:-1] * cp[1:-1])
        rhs[-1] = physics.melting_point(self.H)
        T = solve_banded((1, 1), ab, rhs)
        dz = self.H / (self.nx - 1)
        return T, k_base * (T[-1] - T[-2]) / dz - q_tidal[-1] * dz / 2
