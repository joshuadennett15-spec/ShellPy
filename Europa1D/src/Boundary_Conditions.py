import numpy as np
import numpy.typing as npt
from abc import ABC, abstractmethod
from typing import Tuple, Optional

# Import from modular structure
from constants import Thermal, Planetary, FloatOrArray
from Physics import IcePhysics

"""
Boundary conditions for Europa Ice Shell Model

Utilizes Abstract Base Classes:
- SurfaceCondition (ABC): Blueprint for all surface models
- FixedTemperature: Dirichlet boundary (constant T)
- StefanBoltzmann: Radiative equilibrium with Newton-Raphson solver
"""

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Abstract Base Classes
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

class surfacecondition(ABC):
    """
    Abstract Base Class for all surface boundary conditions
    """

    @abstractmethod
    def get_temperature(self, time: float, current_T: float,
                        k: float, dz: float) -> float:
        """
        Args:
            time: Current simulation time (s)
            current_T: Current surface temperature estimate (K)
            k: Thermal conductivity at surface (W/m·K)
            dz: Grid spacing at surface (m)

        Returns:
            Surface temperature (K)
        """
        pass

    @abstractmethod
    def get_linearization(self, time: float, current_T: float, k: float,
                          dz: float) -> Tuple[float, float, float]:
        """
        Returns coefficients (a, b, c) for implicit Crank-Nicolson coupling.

        The boundary condition is expressed as: a·T_s^{n+1} + b = c·T_1^{n+1}

        Args:
            time: Current simulation time (s)
            current_T: Current surface temperature (K)
            k: Thermal conductivity at surface (W/m·K)
            dz: Grid spacing at surface (m)

        Returns:
            (a, b, c): Coefficients for matrix assembly

        """
        pass

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# IMPLEMENTATION: Fixed Temperature (Dirichlet)
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

class FixedTemperature(surfacecondition):
    """
    Dirichlet boundary condition: Surface temperature is fixed.
    """
    def __init__(self, temperature: float = Thermal.SURFACE_TEMP_MEAN):
        """
        Args:
            temperature: Fixed surface temperature (K)
        """
        self.temperature = temperature

    def get_temperature(self, time: float, current_T: float,
                        k: float, dz: float) -> float:
        """
        Returns the fixed surface temperature.
        """
        return self.temperature

    def get_linearization(self, time: float, current_T: float,
                          k: float, dz: float) -> Tuple[float, float, float]:
        """
        Linearization for Dirichlet: T_s = T_fixed

        In matrix form: 1·T_s + (-T_fixed) = 0·T_1
        """
        return (1.0, -self.temperature, 0.0)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# IMPLEMENTATION: Stefan-Boltzmann Radiative Equilibrium
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

class StefanBoltzmann(surfacecondition):
    """
    Radiative equilibrium boundary condition with solar forcing

    Solves the nonlinear energy balance:
        ε·σ·T⁴ = (1-a)·Q(t) + k·(T₁-T)/dz

    Uses Newton-Raphson iteration for convergence.
    """
    STEFAN_BOLTZMANN: float = Thermal.BOLTZMANN

    MIN_SURFACE_TEMP: float = 20.0              # (k) prevents negative temps in iterator
    INITIAL_GUESS_TEMP: float = 104             # (k) starting temp for Newton-Raphson

    def __init__(self, emissivity: float = 0.94, albedo: float = 0.64,
                 solar_flux: float = 50.5, latitude: float = 0.0, max_iterations: int = 20,
                 tolerance: float = 0.01):
        """

        :param emissivity: Surface emissivity (dimensionless)
        :param albedo: Bond albedo (dimensionless)
        :param solar_flux: Mean solar flux at Europa(W/m²)
        :param latitude: Latitude from equator (degrees)
        :param max_iterations: Newton-Raphson iteration limit
        :param tolerance: Convergence tolerance (k)
        """
        self.emissivity = emissivity
        self.albedo = albedo
        self.solar_flux = solar_flux
        self.latitude_rad = np.radians(latitude)
        self.max_iterations = max_iterations
        self.tolerance = tolerance

    def get_solar_flux(self, time: float, diurnal: bool = False) -> float:
        """
        Returns the solar flux Q(t)
        :param time: Simulation time (s)
        :param diurnal: If True, include diurnal variation
        :return:
            Solar flux (W/m²)
        """
        if not diurnal:
            # Time-averaged flux: Q_mean = Q_solar · cos(lat) / π
            return self.solar_flux * np.cos(self.latitude_rad) / np.pi
        else:
            # Diurnal variation
            rotation_period = Planetary.ROTATION_PERIOD
            hour_angle = 2 * np.pi * (time % rotation_period) / rotation_period
            cos_zenith = np.cos(self.latitude_rad) * np.cos(hour_angle - np.pi)

            return self.solar_flux * max(0.0, cos_zenith)

    def get_temperature(self, time: float, current_T: float, k: float, dz: float) -> float:
        """
        Solves Stefan-Boltzmann BC using Newton-Raphson iteration.

        Energy balance: ε·σ·T⁴ = (1-a)·Q + conductive_flux
        """
        Q_absorbed = (1.0 - self.albedo) * self.get_solar_flux(time)

        # Pure radiative equilibrium if missing parameters
        if current_T is None or k is None or dz is None:
            T_eq = (Q_absorbed / (self.emissivity * self.STEFAN_BOLTZMANN)) ** 0.25

            return T_eq

        # Newton-Raphson iteration
        T_s = current_T if current_T > 0 else self.INITIAL_GUESS_TEMP

        for _ in range(self.max_iterations):
            # radiative flux out
            F_rad = self.emissivity * self.STEFAN_BOLTZMANN * T_s ** 4

            # Residual (assuming local radiative balance)
            residual = F_rad - Q_absorbed

            # derivative: dR/dT_s = 4·ε·σ·T³
            derivative = 4.0 * self.emissivity * self.STEFAN_BOLTZMANN * T_s ** 3

            # Newton update with floor
            delta_T = -residual / derivative
            T_s_new =  max(T_s + delta_T, self.MIN_SURFACE_TEMP)

            if abs(T_s_new - T_s) < self.tolerance:
                return T_s_new

            T_s = T_s_new

        return T_s

    def get_linearization(self, time: float, current_T: float, k: float,
                          dz: float) -> Tuple[float, float, float]:
        """
        Returns linearized coefficients for implicit Crank-Nicolson coupling.

        Linearizes T⁴ around current_T:
            T⁴ ≈ 4·T³·T_new - 3·T⁴
        """

        Q_absorbed = (1.0 - self.albedo) * self.get_solar_flux(time)
        eps_sig = self.emissivity * self.STEFAN_BOLTZMANN

        # From: ε·σ·(4·T³·T_new - 3·T⁴) = Q + k·(T₁-T_new)/dz
        # Rearrange: (4·ε·σ·T³ + k/dz)·T_new = Q + 3·ε·σ·T⁴ + k·T₁/dz

        a = 4.0 * eps_sig * current_T ** 3 + k / dz
        b = -(Q_absorbed + 3.0 * eps_sig * current_T ** 4)
        c = k / dz

        return (a, b, c)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# IMPLEMENTATION: Equilibrium Checking
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def check_equilibrium(db_dt: float, threshold: float = 1e-15) -> bool:
    """
    Determines if the shell has reached thermal equilibrium (db/dt → 0).
    Args:
        db_dt: Current freezing front velocity (m/s)
        threshold: Velocity threshold for equilibrium (m/s)

    Returns:
        True if equilibrium reached
    """
    return abs(db_dt) < threshold

