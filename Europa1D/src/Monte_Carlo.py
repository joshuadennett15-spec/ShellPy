"""
Monte Carlo Uncertainty Quantification for Europa Ice Shell Thickness

Implements the Monte Carlo simulation framework following Howell (2021)
methodology to generate probability density distributions for ice shell thickness.

Features:
    - Full transient thermal evolution using ThermalSolver
    - Multiprocessing parallelization across CPU cores
    - Configurable parameter sampling via ParameterSampler
    - Savitzky-Golay smoothing for peak identification
    - Statistical analysis (CBE, σ bounds, convergence tracking)

Architecture:
    Uses Protocol pattern for parameter sampling to allow custom distributions.
    MonteCarloRunner orchestrates parallel execution.
    MonteCarloResults is a frozen dataclass for immutable results.

References:
    - Howell (2021): Monte Carlo methodology and parameter distributions
"""

from runtime_support import configure_numeric_runtime, resolve_worker_count

configure_numeric_runtime()

import numpy as np
import numpy.typing as npt
from typing import Dict, Optional, Protocol, Callable, Any, Tuple
from dataclasses import dataclass, field
import time
import multiprocessing as mp
from functools import partial

from ConfigManager import ConfigManager
cfg = ConfigManager()

# Import from new modular structure
from Solver import Thermal_Solver
from Physics import IcePhysics
from Boundary_Conditions import FixedTemperature, StefanBoltzmann
from constants import Thermal, Planetary, Rheology, HeatFlux, Convection as ConvConst


# =============================================================================
# TYPE DEFINITIONS
# =============================================================================

class ParameterSampler(Protocol):
    """Protocol for parameter sampling strategies."""

    def sample(self) -> Dict[str, float]:
        """Sample a complete set of parameters for one simulation."""
        ...


# =============================================================================
# RESULTS CONTAINER
# =============================================================================

@dataclass(frozen=True)
class SubpopulationStats:
    """
    Statistics for a convective or conductive subpopulation.

    Following Mitri & Showman (2005), the MC ensemble splits into two
    equilibrium branches: a convective branch (Ra >= Ra_crit, thinner shells)
    and a conductive branch (Ra < Ra_crit, thicker shells).

    Attributes:
        label: 'convective' or 'conductive'
        n_samples: Number of samples in this subpopulation
        fraction: Fraction of total valid samples
        thicknesses_km: Raw thickness array for this subpopulation
        cbe_km: Mode of the subpopulation PDF
        median_km: 50th percentile
        mean_km: Arithmetic mean
        sigma_1_low_km: 15.87th percentile
        sigma_1_high_km: 84.13th percentile
        mean_D_cond_km: Mean conductive lid thickness
        mean_D_conv_km: Mean convective layer thickness
        mean_lid_fraction: Mean D_cond / H
        mean_Ra: Mean Rayleigh number
        mean_Nu: Mean Nusselt number
    """
    label: str
    n_samples: int
    fraction: float
    thicknesses_km: npt.NDArray[np.float64]
    cbe_km: float
    median_km: float
    mean_km: float
    sigma_1_low_km: float
    sigma_1_high_km: float
    mean_D_cond_km: float
    mean_D_conv_km: float
    mean_lid_fraction: float
    mean_Ra: float
    mean_Nu: float


@dataclass(frozen=True)
class MonteCarloResults:
    """
    Immutable container for Monte Carlo simulation results.

    Attributes:
        thicknesses_km: Valid thickness samples (km)
        n_iterations: Total iterations run
        n_valid: Number of valid (physical) solutions
        cbe_km: Current Best Estimate (mode of PDF)
        median_km: Median thickness
        mean_km: Mean thickness
        sigma_1_low_km: 1σ lower bound (16th percentile)
        sigma_1_high_km: 1σ upper bound (84th percentile)
        runtime_seconds: Total simulation runtime
        D_cond_km: Conductive lid thickness for each sample (km)
        D_conv_km: Convective layer thickness for each sample (km)
        lid_fractions: Conductive lid fraction (D_cond/H) for each sample
        Ra_values: Rayleigh number for each sample
        Nu_values: Nusselt number for each sample
    """
    thicknesses_km: npt.NDArray[np.float64]
    n_iterations: int
    n_valid: int
    histogram_bins: npt.NDArray[np.float64]
    histogram_counts: npt.NDArray[np.float64]
    pdf_smoothed: npt.NDArray[np.float64]
    bin_centers: npt.NDArray[np.float64]
    cbe_km: float
    median_km: float
    mean_km: float
    sigma_1_low_km: float
    sigma_1_high_km: float
    runtime_seconds: float
    sampled_params: Optional[Dict[str, npt.NDArray[np.float64]]] = None
    convergence_n: Optional[npt.NDArray[np.float64]] = None
    convergence_mean: Optional[npt.NDArray[np.float64]] = None
    convergence_std: Optional[npt.NDArray[np.float64]] = None
    # Convection diagnostics
    D_cond_km: Optional[npt.NDArray[np.float64]] = None
    D_conv_km: Optional[npt.NDArray[np.float64]] = None
    lid_fractions: Optional[npt.NDArray[np.float64]] = None
    Ra_values: Optional[npt.NDArray[np.float64]] = None
    Nu_values: Optional[npt.NDArray[np.float64]] = None
    # Subpopulation analysis (Mitri & Showman 2005 bistability)
    subpopulations: Optional[Dict[str, SubpopulationStats]] = None
    # Which Nu(Ra) scaling law produced these results
    nu_scaling: str = "green"
    # Scientifically material run choices (for reproducibility)
    run_metadata: Optional[Dict[str, str]] = None


# =============================================================================
# DEFAULT SOLVER CONFIGURATION
# =============================================================================

@dataclass
class SolverConfig:
    """Configuration for each Monte Carlo solver run."""
    nx: int = 31
    initial_thickness: float = 20e3  # Updated: 20 km (closer to expected equilibrium)
    dt: float = 1e12
    total_time: float = 5e14
    eq_threshold: float = 1e-12
    max_steps: int = 1500
    use_convection: bool = True
    rannacher_steps: int = 4
    coordinate_system: str = 'auto'
    use_warm_start: bool = True  # Use physics-based initial guess
    reject_subcritical: bool = False  # Subcritical shells are valid conductive solutions
                                     # False: keep as conductive (polar)
    
    def __post_init__(self):
        # Override with JSON configs if present
        self.use_convection = cfg.get("monte_carlo", "use_convection", self.use_convection)
        self.max_steps = cfg.get("monte_carlo", "max_steps", self.max_steps)


# =============================================================================
# DEFAULT PARAMETER SAMPLER
# =============================================================================

class HowellParameterSampler:
    """
    Parameter sampler following Howell (2021) distributions.

    Uses the exact distribution types and parameters from Table 1 of Howell (2021):
    - Log-Normal: d_grain, f_salt
    - Normal: T_surf, D_H2O, Q_v, Q_b, epsilon_0, H_rad
    - Truncated Normal: mu_ice
    - Uniform: f_porosity

    Note: q_ocean is not sampled directly - it's derived from tidal + radiogenic heating.
    """

    def __init__(self, seed: Optional[int] = None):
        """Initialize with optional random seed."""
        self.rng = np.random.default_rng(seed)

    def _sample_lognormal(self, mean: float, sigma_orders: float) -> float:
        """
        Sample from log-normal with mean and width in orders of magnitude.

        Args:
            mean: Mean value in linear space
            sigma_orders: Width in orders of magnitude (1 = factor of 10)
        """
        mu_log = np.log10(mean)
        sigma_log = sigma_orders / 3.0  # Convert to ~1σ in log space
        return 10 ** self.rng.normal(mu_log, sigma_log)

    def _sample_truncated_normal(self, mean: float, sigma: float,
                                 low: float = -np.inf, high: float = np.inf) -> float:
        """Sample from truncated normal distribution."""
        while True:
            sample = self.rng.normal(mean, sigma)
            if low <= sample <= high:
                return sample

    def sample(self) -> Dict[str, float]:
        """
        Sample a complete set of parameters for one simulation.

        Returns:
            Dictionary of parameter name -> sampled value

        Distributions (Howell 2021 Table 1):
            d_grain:    Log-Normal, mean 0.7 mm, ±0.5 orders of magnitude
            epsilon_0:  Log-Normal, mean 1.2×10⁻⁵, ±0.3 orders of magnitude
            T_surf:     Normal, 104±7 K
            D_H2O:      Normal, 127±21 km
            mu_ice:     Truncated Normal, mean 3.5 GPa, σ=0.5 GPa, [2.0, 5.0] GPa
            Q_v:        Normal, 59.4 kJ/mol (σ = 5%)
            Q_b:        Normal, 49.0 kJ/mol (σ = 5%)
            H_rad:      Normal, 4.5±1.0 pW/kg
            f_porosity: Uniform, 0-30%
            f_salt:     Log-Normal, mean 3%, ±1 order of magnitude
        """
        # Distributions scaled to reproduce Howell 2021 Table 1
        # Grain size: Howell (2021) mean = 0.7 mm, ±0.5 orders of magnitude
        d_grain = 10 ** self.rng.normal(np.log10(7e-4), 0.5)  # 0.7 mm ± 0.5 orders
        d_grain = np.clip(d_grain, 1e-5, 5e-3)  # Physical bounds: 10 μm to 5 mm

        # Strain amplitude: log-normal distribution spanning realistic range
        # Howell's 1.2e-5 strain amplitude baseline
        epsilon_0 = 10 ** self.rng.normal(np.log10(1.2e-5), 0.3)  # ±0.3 orders (1σ)

        # MEDIUM sensitivity parameters
        T_surf = self.rng.normal(104.0, 7.0)  # 104±7 K
        D_H2O = self.rng.normal(127e3, 21e3)  # 127±21 km
        mu_ice = self._sample_truncated_normal(3.5e9, 0.5e9, low=2.0e9, high=5.0e9)

        # LOW sensitivity parameters
        Q_v = self.rng.normal(59.4e3, 0.05 * 59.4e3)  # 59.4 kJ/mol, σ=5%
        Q_b = self.rng.normal(49.0e3, 0.05 * 49.0e3)  # 49.0 kJ/mol, σ=5%
        H_rad = self.rng.normal(4.5e-12, 1.0e-12)  # 4.5±1.0 pW/kg → W/kg
        T_phi = self.rng.normal(150.0, 20.0 / 3.0)  # 150±20 K (3σ)

        # SILICATE TIDAL POWER
        # Log-Normal: Standard Howell distribution features ~100 GW mean
        mean_log = np.log(100e9)  # 100 GW in Watts
        sigma_log = np.log(10) / 3  # 3σ = factor of 10
        P_tidal = self.rng.lognormal(mean=mean_log, sigma=sigma_log)

        # NEGLIGIBLE sensitivity parameters
        f_porosity = self.rng.uniform(0.0, 0.30)  # 0-30%
        f_salt = self._sample_lognormal(0.03, 1.0)  # 3% ± 1 order of magnitude
        f_salt = np.clip(f_salt, 0.0, 0.5)  # Physical limit
        B_k = 10 ** self.rng.uniform(-1.0, 1.0)  # 0.1 to 10 (Howell 2021)

        # Rheology diffusion prefactors and grain boundary width (Howell 2021)
        D0v = self.rng.normal(9.1e-4, 0.033 * 9.1e-4)
        D0b = self.rng.normal(8.4e-4, 0.033 * 8.4e-4)
        d_del_mean = np.mean([9.04e-10, 5.22e-10])
        d_del_std = np.std([9.04e-10, 5.22e-10])
        d_del = self.rng.normal(d_del_mean, d_del_std)

        # Ensure physical bounds
        T_surf = np.clip(T_surf, 50.0, 150.0)
        D_H2O = np.clip(D_H2O, 80e3, 200e3)
        epsilon_0 = np.clip(epsilon_0, 1e-7, 1e-3)
        T_phi = np.clip(T_phi, T_surf + 1.0, Thermal.MELT_TEMP - 1.0)
        D0v = max(D0v, 1e-8)
        D0b = max(D0b, 1e-8)
        d_del = max(d_del, 1e-12)

        return {
            # Rheology (HIGH)
            'd_grain': d_grain,
            'd_del': d_del,
            'D0v': D0v,
            'D0b': D0b,

            # Tidal (HIGH via tidal flux)
            'epsilon_0': epsilon_0,
            'mu_ice': mu_ice,

            # Surface/Structure (MEDIUM)
            'T_surf': T_surf,
            'D_H2O': D_H2O,

            # Activation energies (LOW)
            'Q_v': Q_v,
            'Q_b': Q_b,

            # Radiogenic (LOW)
            'H_rad': H_rad,  # W/kg, needs to be multiplied by silicate mass

            # Silicate Tidal (Howell 2021)
            'P_tidal': P_tidal,  # W, total silicate tidal power

            # Porosity/Salt (NEGLIGIBLE)
            'f_porosity': f_porosity,
            'f_salt': f_salt,
            'T_phi': T_phi,
            'B_k': B_k,
        }

    def set_seed(self, seed: int) -> None:
        """Reset the random state with a new seed."""
        self.rng = np.random.default_rng(seed)


# =============================================================================
# WORKER FUNCTION (FOR MULTIPROCESSING)
# =============================================================================

def _run_single_sample(
        sample_id: int,
        base_seed: int,
        config: SolverConfig,
        sampler_class: type,
) -> Optional[Dict[str, Any]]:
    """
    Run a single Monte Carlo sample using ThermalSolver.

    Designed to be called by multiprocessing.Pool.

    Args:
        sample_id: Index of this sample
        base_seed: Base random seed (combined with sample_id)
        config: Solver configuration
        sampler_class: Class for parameter sampling

    Returns:
        Dict with 'thickness_km' and 'params', or None if non-physical
    """
    try:
        # Create sampler with unique seed
        sampler = sampler_class(seed=base_seed + sample_id)
        params = sampler.sample()

        # Extract key parameters
        T_surf = params['T_surf']
        D_H2O = params['D_H2O']
        epsilon_0 = params['epsilon_0']
        mu_ice = params['mu_ice']
        H_rad = params['H_rad']
        P_tidal = params['P_tidal']

        # =====================================================================
        # CALCULATE OCEAN HEAT FLUX FROM SAMPLED PARAMETERS (Howell 2021)
        # =====================================================================

        # Geometry constants
        R_europa = Planetary.RADIUS  # m
        R_rock = R_europa - D_H2O    # Radius of rocky core (m)
        A_surface = Planetary.AREA   # m² (Europa surface area)
        rho_rock = 3500.0            # kg/m³ (silicate density)

        # Rock mass from sampled D_H2O (Howell 2021 uses rocky body mass)
        M_rock = (4.0 / 3.0) * np.pi * (R_rock ** 3) * rho_rock

        # 1. Radiogenic heat flux: q_rad = H_rad × M_rock / A_surface
        q_radiogenic = (H_rad * M_rock) / A_surface  # W/m²

        # 2. Silicate tidal flux (Howell Eq 6): q_tidal = P_tidal / A_surface
        q_silicate_tidal = P_tidal / A_surface  # W/m²

        # Total basal flux from ocean (sum of radiogenic + silicate tidal)
        q_basal = q_radiogenic + q_silicate_tidal

        # =====================================================================
        # WARM START: Physics-based initial thickness guess
        # =====================================================================
        # Steady-state conductive approximation: H ≈ k·ΔT / q_basal
        if config.use_warm_start and q_basal > 0:
            T_melt = Thermal.MELT_TEMP
            delta_T = T_melt - T_surf
            k_mean = Thermal.conductivity((T_surf + T_melt) / 2)

            H_guess = (k_mean * delta_T) / q_basal
            
            # Convective shells are much thicker than pure conduction predicts
            # because convection (Nu > 1) enhances internal heat transport.
            if config.use_convection:
                H_guess *= 8.0  # Empirical approximation (Nu varies ~3 to 20)

            # Clamp to reasonable range [5 km, 100 km]
            H_guess = np.clip(H_guess, 5e3, 100e3)
        else:
            H_guess = config.initial_thickness

        # Create surface boundary condition
        surface_bc = FixedTemperature(temperature=T_surf)

        # Create solver with warm-start thickness and SAMPLED PHYSICS PARAMS
        solver = Thermal_Solver(
            nx=config.nx,
            thickness=H_guess,
            dt=config.dt,
            total_time=config.total_time,
            coordinate_system=config.coordinate_system,
            surface_bc=surface_bc,
            rannacher_steps=config.rannacher_steps,
            use_convection=config.use_convection,
            physics_params=params,  # CRITICAL: Pass all sampled parameters!
        )

        # Run to equilibrium
        # Note: The solver computes tidal heating internally based on T profile
        # We pass q_basal as the basal flux to balance
        for step in range(config.max_steps):
            velocity = solver.solve_step(q_basal)

            if abs(velocity) < config.eq_threshold:
                break

        # Get final thickness
        H_km = solver.H / 1000.0

        # =====================================================================
        # PHYSICAL FILTERS (Howell 2021)
        # =====================================================================

        # Filter 1: Must be positive and below hydrosphere limit
        D_H2O_km = D_H2O / 1000.0
        if H_km <= 0.5 or H_km >= D_H2O_km * 0.99:
            return None

        # Filter 2: Reasonable upper limit
        if H_km > 200:
            return None

        # (Filter 3 removed — lid stability is handled by the solver's
        #  convection parameterization and Filter 4 below.)
        from Convection import IceConvection
        from constants import Convection as ConvectionConstants

        # Filter 4: Check convective sublayer criticality
        subcritical = False
        if config.use_convection and solver.convection_state is not None:
            state = solver.convection_state
            if state.D_conv > 0 and state.Ra < ConvectionConstants.RA_CRIT:
                if config.reject_subcritical:
                    # Equator/global: reject unphysical thick-shell subcritical
                    return None
                else:
                    # Polar: keep as purely conductive
                    subcritical = True

        # =====================================================================
        # CAPTURE CONVECTION DIAGNOSTICS
        # =====================================================================
        if subcritical or solver.convection_state is None:
            # Purely conductive: entire shell is the lid
            conv_diag = {
                'D_cond_km': H_km,
                'D_conv_km': 0.0,
                'lid_fraction': 1.0,
                'Ra': solver.convection_state.Ra if solver.convection_state else 0.0,
                'Nu': 1.0,
            }
        else:
            state = solver.convection_state
            conv_diag = {
                'D_cond_km': state.D_cond / 1000.0,
                'D_conv_km': state.D_conv / 1000.0,
                'lid_fraction': state.D_cond / solver.H if solver.H > 0 else 1.0,
                'Ra': state.Ra,
                'Nu': state.Nu,
            }

        return {'thickness_km': H_km, 'params': params, 'convection': conv_diag}

    except Exception:
        return None


# =============================================================================
# MONTE CARLO RUNNER
# =============================================================================

class MonteCarloRunner:
    """
    Monte Carlo simulation runner for Europa ice shell thickness.

    Uses ThermalSolver for each sample with multiprocessing parallelization.

    Example:
        runner = MonteCarloRunner(n_iterations=1000, seed=42)
        results = runner.run()
        print(f"CBE: {results.cbe_km:.1f} km")
    """

    def __init__(
            self,
            n_iterations: int = 100,
            seed: Optional[int] = None,
            verbose: bool = True,
            n_workers: Optional[int] = None,
            config: Optional[SolverConfig] = None,
            sampler_class: type = HowellParameterSampler,
    ):
        """
        Initialize the Monte Carlo runner.

        Args:
            n_iterations: Number of Monte Carlo samples
            seed: Random seed for reproducibility
            verbose: Print progress updates
            n_workers: Number of parallel workers (None = auto)
            config: Solver configuration (None = defaults)
            sampler_class: Class for parameter sampling
        """
        self.n_iterations = n_iterations
        self.seed = seed if seed is not None else int(time.time())
        self.verbose = verbose
        self.n_workers = resolve_worker_count(n_workers)
        self.config = config or SolverConfig()
        self.sampler_class = sampler_class

    def run(self) -> MonteCarloResults:
        """
        Execute the Monte Carlo simulation.

        Returns:
            MonteCarloResults with all statistics and raw data
        """
        if self.verbose:
            self._print_header()

        start_time = time.time()

        # Create worker function with fixed arguments
        worker = partial(
            _run_single_sample,
            base_seed=self.seed,
            config=self.config,
            sampler_class=self.sampler_class,
        )

        # Run in parallel or sequential
        results = self._run_parallel(worker) if self.n_workers > 1 else self._run_sequential(worker)

        runtime = time.time() - start_time

        # Extract valid results
        valid_results = [r for r in results if r is not None]

        if len(valid_results) == 0:
            raise RuntimeError("No valid solutions. Check parameter distributions.")

        thicknesses = np.array([r['thickness_km'] for r in valid_results])

        # Build sampled params dictionary
        sampled_params = self._extract_sampled_params(valid_results)

        # Extract convection diagnostics
        convection_data = self._extract_convection_data(valid_results)

        # Compute convergence statistics
        conv_n, conv_mean, conv_std = self._compute_convergence(thicknesses)

        # Analyze and return results
        mc_results = self._analyze_results(
            thicknesses, runtime, sampled_params, conv_n, conv_mean, conv_std,
            convection_data
        )

        if self.verbose:
            self._print_summary(mc_results)

        return mc_results

    def _run_parallel(self, worker: Callable) -> list:
        """Run samples in parallel using multiprocessing."""
        with mp.Pool(self.n_workers) as pool:
            chunksize = max(1, self.n_iterations // (self.n_workers * 4))
            results = []

            for i, result in enumerate(
                    pool.imap_unordered(worker, range(self.n_iterations), chunksize=chunksize)
            ):
                results.append(result)
                self._log_progress(i + 1, results)

        return results

    def _run_sequential(self, worker: Callable) -> list:
        """Run samples sequentially (fallback for single worker)."""
        results = []
        for i in range(self.n_iterations):
            results.append(worker(i))
            self._log_progress(i + 1, results)
        return results

    def _log_progress(self, completed: int, results: list) -> None:
        """Log progress if verbose and at checkpoint."""
        if self.verbose and completed % max(1, self.n_iterations // 10) == 0:
            pct = 100 * completed / self.n_iterations
            valid = sum(1 for r in results if r is not None)
            print(f"  Progress: {pct:5.1f}% | Valid: {valid}/{completed}")

    def _extract_sampled_params(self, valid_results: list) -> Dict[str, npt.NDArray]:
        """Extract sampled parameters as arrays."""
        if not valid_results:
            return {}

        param_names = valid_results[0]['params'].keys()
        return {
            name: np.array([r['params'][name] for r in valid_results])
            for name in param_names
        }

    def _extract_convection_data(self, valid_results: list) -> Dict[str, npt.NDArray]:
        """Extract convection diagnostics as arrays."""
        if not valid_results:
            return {}

        # Check if convection data exists
        if 'convection' not in valid_results[0]:
            return {}

        return {
            'D_cond_km': np.array([r['convection']['D_cond_km'] for r in valid_results]),
            'D_conv_km': np.array([r['convection']['D_conv_km'] for r in valid_results]),
            'lid_fraction': np.array([r['convection']['lid_fraction'] for r in valid_results]),
            'Ra': np.array([r['convection']['Ra'] for r in valid_results]),
            'Nu': np.array([r['convection']['Nu'] for r in valid_results]),
        }

    def _compute_convergence(
            self, thicknesses: npt.NDArray
    ) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        """Compute running mean/std at log-spaced checkpoints."""
        n_valid = len(thicknesses)

        if n_valid < 20:
            checkpoints = np.arange(1, n_valid + 1)
        else:
            checkpoints = np.unique(
                np.logspace(1, np.log10(n_valid), 50).astype(int)
            )
            checkpoints = checkpoints[checkpoints <= n_valid]

        means = np.array([np.mean(thicknesses[:n]) for n in checkpoints])
        stds = np.array([np.std(thicknesses[:n]) / np.sqrt(n) for n in checkpoints])

        return checkpoints, means, stds

    @staticmethod
    def _compute_subpopulation(
            label: str,
            mask: npt.NDArray[np.bool_],
            thicknesses: npt.NDArray,
            n_total: int,
            convection_data: Optional[Dict[str, npt.NDArray]],
    ) -> SubpopulationStats:
        """Compute statistics for a single subpopulation (convective or conductive)."""
        H_sub = thicknesses[mask]
        n_sub = len(H_sub)

        if n_sub == 0:
            return SubpopulationStats(
                label=label, n_samples=0, fraction=0.0,
                thicknesses_km=H_sub,
                cbe_km=0.0, median_km=0.0, mean_km=0.0,
                sigma_1_low_km=0.0, sigma_1_high_km=0.0,
                mean_D_cond_km=0.0, mean_D_conv_km=0.0,
                mean_lid_fraction=0.0, mean_Ra=0.0, mean_Nu=0.0,
            )

        # Mode via KDE (needs >= 2 samples)
        if n_sub >= 2:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(H_sub)
            x_grid = np.linspace(H_sub.min(), H_sub.max(), 200)
            cbe = float(x_grid[np.argmax(kde(x_grid))])
        else:
            cbe = float(H_sub[0])

        D_cond = convection_data['D_cond_km'][mask] if convection_data else np.zeros(n_sub)
        D_conv = convection_data['D_conv_km'][mask] if convection_data else np.zeros(n_sub)
        lid_f = convection_data['lid_fraction'][mask] if convection_data else np.ones(n_sub)
        Ra = convection_data['Ra'][mask] if convection_data else np.zeros(n_sub)
        Nu = convection_data['Nu'][mask] if convection_data else np.ones(n_sub)

        return SubpopulationStats(
            label=label,
            n_samples=n_sub,
            fraction=n_sub / n_total,
            thicknesses_km=H_sub,
            cbe_km=cbe,
            median_km=float(np.median(H_sub)),
            mean_km=float(np.mean(H_sub)),
            sigma_1_low_km=float(np.percentile(H_sub, 15.87)),
            sigma_1_high_km=float(np.percentile(H_sub, 84.13)),
            mean_D_cond_km=float(np.mean(D_cond)),
            mean_D_conv_km=float(np.mean(D_conv)),
            mean_lid_fraction=float(np.mean(lid_f)),
            mean_Ra=float(np.mean(Ra)),
            mean_Nu=float(np.mean(Nu)),
        )

    def _split_subpopulations(
            self,
            thicknesses: npt.NDArray,
            convection_data: Optional[Dict[str, npt.NDArray]],
    ) -> Optional[Dict[str, SubpopulationStats]]:
        """
        Split MC results into convective and conductive subpopulations.

        Classification: Ra >= Ra_crit → convective, Ra < Ra_crit → conductive.
        Following Mitri & Showman (2005) bistability framework.
        """
        if convection_data is None or 'Ra' not in convection_data:
            return None

        Ra = convection_data['Ra']
        ra_crit = ConvConst.RA_CRIT
        n_total = len(thicknesses)

        convective_mask = Ra >= ra_crit
        conductive_mask = ~convective_mask

        return {
            'convective': self._compute_subpopulation(
                'convective', convective_mask, thicknesses, n_total, convection_data),
            'conductive': self._compute_subpopulation(
                'conductive', conductive_mask, thicknesses, n_total, convection_data),
        }

    def _analyze_results(
            self,
            thicknesses: npt.NDArray,
            runtime: float,
            sampled_params: Dict[str, npt.NDArray],
            conv_n: npt.NDArray,
            conv_mean: npt.NDArray,
            conv_std: npt.NDArray,
            convection_data: Optional[Dict[str, npt.NDArray]] = None,
    ) -> MonteCarloResults:
        """Compute histogram, PDF, and statistics."""
        n_valid = len(thicknesses)

        # Freedman-Diaconis bin width
        iqr = np.percentile(thicknesses, 75) - np.percentile(thicknesses, 25)
        if iqr > 0:
            bin_width = 2 * iqr / (n_valid ** (1 / 3))
            n_bins = int(np.ceil((thicknesses.max() - thicknesses.min()) / bin_width))
            n_bins = np.clip(n_bins, 10, 100)
        else:
            n_bins = 20

        counts, bin_edges = np.histogram(thicknesses, bins=n_bins, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Kernel Density Estimation (KDE) smoothing
        from scipy.stats import gaussian_kde
        if len(thicknesses) > 1:
            kde = gaussian_kde(thicknesses)
            pdf_smoothed = kde(bin_centers)
            # Find a more precise Mode (CBE) using dense grid
            x_grid = np.linspace(thicknesses.min(), thicknesses.max(), 300)
            cbe = float(x_grid[np.argmax(kde(x_grid))])
        else:
            pdf_smoothed = np.zeros_like(bin_centers)
            cbe = float(bin_centers[np.argmax(counts)]) if len(counts) > 0 else 0.0

        # Extract convection arrays (or None if not available)
        D_cond = convection_data.get('D_cond_km') if convection_data else None
        D_conv = convection_data.get('D_conv_km') if convection_data else None
        lid_fracs = convection_data.get('lid_fraction') if convection_data else None
        Ra_vals = convection_data.get('Ra') if convection_data else None
        Nu_vals = convection_data.get('Nu') if convection_data else None

        # Split into convective/conductive subpopulations (Mitri & Showman 2005)
        subpops = self._split_subpopulations(thicknesses, convection_data)

        return MonteCarloResults(
            thicknesses_km=thicknesses,
            n_iterations=self.n_iterations,
            n_valid=n_valid,
            histogram_bins=bin_edges,
            histogram_counts=counts,
            pdf_smoothed=pdf_smoothed,
            bin_centers=bin_centers,
            cbe_km=cbe,
            median_km=float(np.percentile(thicknesses, 50)),
            mean_km=float(np.mean(thicknesses)),
            sigma_1_low_km=float(np.percentile(thicknesses, 15.87)),
            sigma_1_high_km=float(np.percentile(thicknesses, 84.13)),
            runtime_seconds=runtime,
            sampled_params=sampled_params,
            convergence_n=conv_n,
            convergence_mean=conv_mean,
            convergence_std=conv_std,
            D_cond_km=D_cond,
            D_conv_km=D_conv,
            lid_fractions=lid_fracs,
            Ra_values=Ra_vals,
            Nu_values=Nu_vals,
            subpopulations=subpops,
            nu_scaling=ConvConst.NU_SCALING,
            run_metadata={
                "nu_scaling": ConvConst.NU_SCALING,
                "ra_crit": str(ConvConst.RA_CRIT),
                "sampler_class": self.sampler_class.__name__,
                "seed": str(self.seed),
                "nx": str(self.config.nx),
                "use_convection": str(self.config.use_convection),
                "reject_subcritical": str(self.config.reject_subcritical),
            },
        )



    def _print_header(self) -> None:
        """Print simulation header."""
        print("=" * 60)
        print("MONTE CARLO SIMULATION: Europa Ice Shell Thickness")
        print("=" * 60)
        print(f"Iterations: {self.n_iterations}")
        print(f"Workers: {self.n_workers}")
        print(f"Solver: ThermalSolver (nx={self.config.nx}, dt={self.config.dt:.1e}s)")
        print("-" * 60)

    def _print_summary(self, results: MonteCarloResults) -> None:
        """Print results summary."""
        print("-" * 60)
        print("RESULTS SUMMARY")
        print("-" * 60)
        print(f"  Valid samples:    {results.n_valid} / {results.n_iterations} "
              f"({100 * results.n_valid / results.n_iterations:.1f}%)")
        print(f"  Runtime:          {results.runtime_seconds:.2f} seconds")
        print(f"  Time per sample:  {results.runtime_seconds / results.n_iterations:.2f} seconds")
        print()
        print(f"  COMBINED DISTRIBUTION:")
        print(f"  CBE (mode):       {results.cbe_km:.1f} km")
        print(f"  Median:           {results.median_km:.1f} km")
        print(f"  Mean:             {results.mean_km:.1f} km")
        print(f"  1-sigma range:    [{results.sigma_1_low_km:.1f}, {results.sigma_1_high_km:.1f}] km")

        # Subpopulation breakdown (Mitri & Showman 2005 bistability)
        if results.subpopulations is not None:
            print()
            print("  SUBPOPULATION ANALYSIS (Ra_crit = {:.0f}):".format(ConvConst.RA_CRIT))
            for key in ('convective', 'conductive'):
                sub = results.subpopulations[key]
                if sub.n_samples == 0:
                    print(f"    {key.upper()}: 0 samples")
                    continue
                print(f"    {key.upper()} ({sub.n_samples} samples, {sub.fraction:.1%}):")
                print(f"      CBE: {sub.cbe_km:.1f} km | "
                      f"Median: {sub.median_km:.1f} km | "
                      f"1-sigma: [{sub.sigma_1_low_km:.1f}, {sub.sigma_1_high_km:.1f}] km")
                print(f"      D_cond: {sub.mean_D_cond_km:.1f} km | "
                      f"D_conv: {sub.mean_D_conv_km:.1f} km | "
                      f"Lid: {sub.mean_lid_fraction:.1%}")
                print(f"      Ra: {sub.mean_Ra:.2e} | Nu: {sub.mean_Nu:.1f}")

        # Legacy convection stats (backward compatible)
        elif results.lid_fractions is not None:
            print()
            print("  CONVECTION STRUCTURE:")
            mean_lid = float(np.mean(results.lid_fractions))
            mean_D_cond = float(np.mean(results.D_cond_km))
            mean_D_conv = float(np.mean(results.D_conv_km))
            print(f"  Mean lid fraction:    {mean_lid:.1%}")
            print(f"  Mean D_cond:          {mean_D_cond:.1f} km")
            print(f"  Mean D_conv:          {mean_D_conv:.1f} km")
            if results.Ra_values is not None:
                mean_Ra = float(np.mean(results.Ra_values))
                mean_Nu = float(np.mean(results.Nu_values))
                print(f"  Mean Ra:              {mean_Ra:.2e}")
                print(f"  Mean Nu:              {mean_Nu:.1f}")
        print("=" * 60)


# =============================================================================
# I/O UTILITIES
# =============================================================================

def save_results(results: MonteCarloResults, filepath: str = "monte_carlo_results.npz") -> None:
    """Save Monte Carlo results to NumPy archive."""
    save_dict = {
        'thicknesses_km': results.thicknesses_km,
        'n_iterations': results.n_iterations,
        'n_valid': results.n_valid,
        'histogram_bins': results.histogram_bins,
        'histogram_counts': results.histogram_counts,
        'pdf_smoothed': results.pdf_smoothed,
        'bin_centers': results.bin_centers,
        'cbe_km': results.cbe_km,
        'median_km': results.median_km,
        'mean_km': results.mean_km,
        'sigma_1_low_km': results.sigma_1_low_km,
        'sigma_1_high_km': results.sigma_1_high_km,
        'runtime_seconds': results.runtime_seconds,
    }

    if results.convergence_n is not None:
        save_dict['convergence_n'] = results.convergence_n
        save_dict['convergence_mean'] = results.convergence_mean
        save_dict['convergence_std'] = results.convergence_std

    if results.sampled_params is not None:
        for key, values in results.sampled_params.items():
            save_dict[f'param_{key}'] = values

    # Save convection diagnostics
    if results.D_cond_km is not None:
        save_dict['D_cond_km'] = results.D_cond_km
    if results.D_conv_km is not None:
        save_dict['D_conv_km'] = results.D_conv_km
    if results.lid_fractions is not None:
        save_dict['lid_fractions'] = results.lid_fractions
    if results.Ra_values is not None:
        save_dict['Ra_values'] = results.Ra_values
    if results.Nu_values is not None:
        save_dict['Nu_values'] = results.Nu_values

    save_dict['nu_scaling'] = results.nu_scaling

    if results.run_metadata is not None:
        for key, val in results.run_metadata.items():
            save_dict[f'meta_{key}'] = val

    np.savez(filepath, **save_dict)
    print(f"Results saved to: {filepath}")


def load_results(filepath: str = "monte_carlo_results.npz") -> MonteCarloResults:
    """Load Monte Carlo results from NumPy archive."""
    data = np.load(filepath)

    # Load optional fields
    conv_n = data['convergence_n'] if 'convergence_n' in data else None
    conv_mean = data['convergence_mean'] if 'convergence_mean' in data else None
    conv_std = data['convergence_std'] if 'convergence_std' in data else None

    # Load sampled params
    sampled_params = {}
    for key in data.keys():
        if key.startswith('param_'):
            sampled_params[key[6:]] = data[key]

    # Load convection diagnostics
    D_cond = data['D_cond_km'] if 'D_cond_km' in data else None
    D_conv = data['D_conv_km'] if 'D_conv_km' in data else None
    lid_fracs = data['lid_fractions'] if 'lid_fractions' in data else None
    Ra_vals = data['Ra_values'] if 'Ra_values' in data else None
    Nu_vals = data['Nu_values'] if 'Nu_values' in data else None
    nu_scaling = str(data['nu_scaling']) if 'nu_scaling' in data else "green"

    run_metadata = {}
    for key in data.keys():
        if key.startswith('meta_'):
            run_metadata[key[5:]] = str(data[key])

    return MonteCarloResults(
        thicknesses_km=data['thicknesses_km'],
        n_iterations=int(data['n_iterations']),
        n_valid=int(data['n_valid']),
        histogram_bins=data['histogram_bins'],
        histogram_counts=data['histogram_counts'],
        pdf_smoothed=data['pdf_smoothed'],
        bin_centers=data['bin_centers'],
        cbe_km=float(data['cbe_km']),
        median_km=float(data['median_km']),
        mean_km=float(data['mean_km']),
        sigma_1_low_km=float(data['sigma_1_low_km']),
        sigma_1_high_km=float(data['sigma_1_high_km']),
        runtime_seconds=float(data['runtime_seconds']),
        sampled_params=sampled_params if sampled_params else None,
        convergence_n=conv_n,
        convergence_mean=conv_mean,
        convergence_std=conv_std,
        D_cond_km=D_cond,
        D_conv_km=D_conv,
        lid_fractions=lid_fracs,
        Ra_values=Ra_vals,
        Nu_values=Nu_vals,
        nu_scaling=nu_scaling,
        run_metadata=run_metadata if run_metadata else None,
    )


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import os

    # Required for Windows/macOS multiprocessing safety
    mp.freeze_support()

    RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')

    # Limit workers to avoid Windows memory/DLL issues with scipy multiprocessing
    runner = MonteCarloRunner(n_iterations=5000, seed=43, n_workers=13)
    results = runner.run()
    save_results(results, os.path.join(RESULTS_DIR, "monte_carlo_results.npz"))

    print(f"\nTarget CBE: ~24.3 km (Howell 2021)")
    print(f"Computed CBE: {results.cbe_km:.1f} km")
