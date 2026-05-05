"""
Literature-grounded Sobol workflow for the one-dimensional ice-shell model.

This module implements a true Saltelli/Sobol design for the audited shell
priors. It intentionally separates three pieces of the workflow:

1. Generate a Sobol design in the unit hypercube.
2. Map those unit-cube draws to the audited scientific priors via inverse CDFs.
3. Evaluate the thermal model on the resulting deterministic parameter sets.

Design choices follow the current audited baseline and the Sobol/Saltelli
guidance used by SALib:
  - use a dedicated Sobol design, not arbitrary Monte Carlo draws,
  - focus on first- and total-order indices first,
  - use convergence checkpoints at powers of two,
  - keep the main run on a reduced, interpretable parameter set,
  - track numerical/physical validity explicitly rather than silently dropping
    failed evaluations.
"""
from __future__ import annotations

from dataclasses import dataclass
import multiprocessing as mp
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
from scipy.stats import norm

from constants import Planetary


_PPF_EPS = 1.0e-12
_RHO_ROCK = 3500.0

# Fixed audited baseline values for secondary / de-emphasized parameters.
_DEFAULT_D0V = 9.1e-4
_DEFAULT_D0B = 8.4e-4
_DEFAULT_D_DEL = float(np.mean([9.04e-10, 5.22e-10]))
_DEFAULT_F_SALT = 0.0
_DEFAULT_B_K = 1.0
_DEFAULT_T_PHI = 150.0
_WARM_START_CONVECTION_FACTOR = 8.0


@dataclass(frozen=True)
class SobolScenario:
    name: str
    label: str
    enhancement_factor: float
    t_surf_mean: float
    t_surf_sigma: float
    t_surf_low: float
    t_surf_high: float
    epsilon_mean_log10: float
    epsilon_sigma_log10: float
    epsilon_low: float
    epsilon_high: float


@dataclass(frozen=True)
class PriorSpec:
    name: str
    group: str
    dist: str
    low: float
    high: float
    mean: Optional[float] = None
    sigma: Optional[float] = None


@dataclass(frozen=True)
class SobolMappedSample:
    prior_inputs: Dict[str, float]
    solver_params: Dict[str, float]
    diagnostics: Dict[str, float]


SCENARIOS: Dict[str, SobolScenario] = {
    "global_audited": SobolScenario(
        name="global_audited",
        label="Global Audited",
        enhancement_factor=1.0,
        t_surf_mean=104.0,
        t_surf_sigma=7.0,
        t_surf_low=80.0,
        t_surf_high=120.0,
        epsilon_mean_log10=np.log10(1.2e-5),
        epsilon_sigma_log10=0.3,
        epsilon_low=2.0e-6,
        epsilon_high=3.4e-5,
    ),
    "equatorial_baseline": SobolScenario(
        name="equatorial_baseline",
        label="Equatorial Baseline",
        enhancement_factor=1.0,
        t_surf_mean=110.0,
        t_surf_sigma=5.0,
        t_surf_low=95.0,
        t_surf_high=120.0,
        epsilon_mean_log10=np.log10(6.0e-6),
        epsilon_sigma_log10=0.2,
        epsilon_low=2.0e-6,
        epsilon_high=2.0e-5,
    ),
    "equatorial_moderate": SobolScenario(
        name="equatorial_moderate",
        label="Equatorial Moderate",
        enhancement_factor=1.2,
        t_surf_mean=110.0,
        t_surf_sigma=5.0,
        t_surf_low=95.0,
        t_surf_high=120.0,
        epsilon_mean_log10=np.log10(6.0e-6),
        epsilon_sigma_log10=0.2,
        epsilon_low=2.0e-6,
        epsilon_high=2.0e-5,
    ),
    "equatorial_strong": SobolScenario(
        name="equatorial_strong",
        label="Equatorial Strong",
        enhancement_factor=1.5,
        t_surf_mean=110.0,
        t_surf_sigma=5.0,
        t_surf_low=95.0,
        t_surf_high=120.0,
        epsilon_mean_log10=np.log10(6.0e-6),
        epsilon_sigma_log10=0.2,
        epsilon_low=2.0e-6,
        epsilon_high=2.0e-5,
    ),
}


DEFAULT_ANALYSIS_OUTPUTS: Sequence[str] = (
    "valid_flag",
    "physical_flag",
    "convective_flag",
    "thickness_km",
    "D_cond_km",
    "D_conv_km",
    "lid_fraction",
    "Ra",
    "Nu",
)


def get_sobol_scenario(name: str) -> SobolScenario:
    try:
        return SCENARIOS[name]
    except KeyError as exc:
        valid = ", ".join(sorted(SCENARIOS))
        raise ValueError(f"Unknown Sobol scenario '{name}'. Valid scenarios: {valid}") from exc


def get_primary_prior_specs(scenario_name: str) -> List[PriorSpec]:
    """
    Return the reduced parameter set used for the main Sobol runs.

    The set is intentionally smaller than the raw Howell parameter list. Fixed
    terms remain available in the Monte Carlo code, but the first Sobol pass
    focuses on the parameters that remain scientifically interpretable under the
    audited baseline.
    """
    scenario = get_sobol_scenario(scenario_name)
    return [
        PriorSpec(
            name="q_basal_target_mW_m2",
            group="basal_flux",
            dist="uniform",
            low=5.0,
            high=25.0,
        ),
        PriorSpec(
            name="d_grain_mm",
            group="shell_rheology",
            dist="truncnorm_log10",
            low=0.05,
            high=3.0,
            mean=np.log10(0.6),
            sigma=0.35,
        ),
        PriorSpec(
            name="epsilon_0",
            group="shell_tides",
            dist="truncnorm_log10",
            low=scenario.epsilon_low,
            high=scenario.epsilon_high,
            mean=scenario.epsilon_mean_log10,
            sigma=scenario.epsilon_sigma_log10,
        ),
        PriorSpec(
            name="T_surf_K",
            group="surface_boundary",
            dist="truncnorm",
            low=scenario.t_surf_low,
            high=scenario.t_surf_high,
            mean=scenario.t_surf_mean,
            sigma=scenario.t_surf_sigma,
        ),
        PriorSpec(
            name="D_H2O_km",
            group="surface_boundary",
            dist="truncnorm",
            low=80.0,
            high=200.0,
            mean=127.0,
            sigma=21.0,
        ),
        PriorSpec(
            name="mu_ice_GPa",
            group="shell_rheology",
            dist="truncnorm",
            low=2.0,
            high=5.0,
            mean=3.5,
            sigma=0.5,
        ),
        PriorSpec(
            name="Q_v_kJ_mol",
            group="shell_rheology",
            dist="truncnorm",
            low=45.0,
            high=75.0,
            mean=59.4,
            sigma=0.05 * 59.4,
        ),
        PriorSpec(
            name="Q_b_kJ_mol",
            group="shell_rheology",
            dist="truncnorm",
            low=35.0,
            high=65.0,
            mean=49.0,
            sigma=0.05 * 49.0,
        ),
        PriorSpec(
            name="H_rad_pW_kg",
            group="basal_flux",
            dist="truncnorm",
            low=0.1,
            high=10.0,
            mean=4.5,
            sigma=1.0,
        ),
        PriorSpec(
            name="f_porosity",
            group="porosity",
            dist="uniform",
            low=0.0,
            high=0.10,
        ),
    ]


def ordered_unique(values: Sequence[str]) -> List[str]:
    return list(dict.fromkeys(values))


def effective_dimension(problem: Mapping[str, Any]) -> int:
    if "groups" in problem:
        return len(ordered_unique(problem["groups"]))
    return int(problem["num_vars"])


def expected_sobol_rows(
    problem: Mapping[str, Any], base_sample_size: int, calc_second_order: bool
) -> int:
    d_eff = effective_dimension(problem)
    multiplier = (2 * d_eff + 2) if calc_second_order else (d_eff + 2)
    return int(base_sample_size) * multiplier


def is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def build_salib_problem(scenario_name: str, grouped: bool = False) -> Dict[str, Any]:
    specs = get_primary_prior_specs(scenario_name)
    problem: Dict[str, Any] = {
        "num_vars": len(specs),
        "names": [spec.name for spec in specs],
        "bounds": [[0.0, 1.0] for _ in specs],
    }
    if grouped:
        problem["groups"] = [spec.group for spec in specs]
    return problem


def _clip_unit_interval(u: float) -> float:
    return float(np.clip(u, _PPF_EPS, 1.0 - _PPF_EPS))


def _truncated_normal_ppf(u: float, mean: float, sigma: float, low: float, high: float) -> float:
    u = _clip_unit_interval(u)
    a = norm.cdf((low - mean) / sigma)
    b = norm.cdf((high - mean) / sigma)
    q = a + u * (b - a)
    return float(mean + sigma * norm.ppf(q))


def _apply_prior_transform(unit_value: float, spec: PriorSpec) -> float:
    if spec.dist == "uniform":
        return float(spec.low + _clip_unit_interval(unit_value) * (spec.high - spec.low))
    if spec.dist == "truncnorm":
        if spec.mean is None or spec.sigma is None:
            raise ValueError(f"Prior '{spec.name}' is missing mean/sigma")
        return _truncated_normal_ppf(unit_value, spec.mean, spec.sigma, spec.low, spec.high)
    if spec.dist == "truncnorm_log10":
        if spec.mean is None or spec.sigma is None:
            raise ValueError(f"Prior '{spec.name}' is missing mean/sigma")
        log_value = _truncated_normal_ppf(
            unit_value,
            spec.mean,
            spec.sigma,
            np.log10(spec.low),
            np.log10(spec.high),
        )
        return float(10.0 ** log_value)
    raise ValueError(f"Unsupported prior transform '{spec.dist}'")


def map_unit_sample_to_model(unit_sample: Sequence[float], scenario_name: str) -> SobolMappedSample:
    specs = get_primary_prior_specs(scenario_name)
    if len(unit_sample) != len(specs):
        raise ValueError(
            f"Expected {len(specs)} unit parameters for scenario '{scenario_name}', "
            f"received {len(unit_sample)}"
        )

    prior_inputs = {
        spec.name: _apply_prior_transform(float(u), spec)
        for spec, u in zip(specs, unit_sample)
    }

    q_basal_target = prior_inputs["q_basal_target_mW_m2"] * 1.0e-3
    d_grain = prior_inputs["d_grain_mm"] * 1.0e-3
    D_H2O = prior_inputs["D_H2O_km"] * 1.0e3
    mu_ice = prior_inputs["mu_ice_GPa"] * 1.0e9
    Q_v = prior_inputs["Q_v_kJ_mol"] * 1.0e3
    Q_b = prior_inputs["Q_b_kJ_mol"] * 1.0e3
    H_rad = prior_inputs["H_rad_pW_kg"] * 1.0e-12

    R_rock = Planetary.RADIUS - D_H2O
    M_rock = (4.0 / 3.0) * np.pi * (R_rock ** 3) * _RHO_ROCK
    q_radiogenic = (H_rad * M_rock) / Planetary.AREA
    q_silicate_tidal = max(0.0, q_basal_target - q_radiogenic)

    scenario = get_sobol_scenario(scenario_name)
    q_silicate_tidal *= scenario.enhancement_factor
    P_tidal = q_silicate_tidal * Planetary.AREA

    solver_params = {
        "d_grain": d_grain,
        "d_del": _DEFAULT_D_DEL,
        "D0v": _DEFAULT_D0V,
        "D0b": _DEFAULT_D0B,
        "epsilon_0": prior_inputs["epsilon_0"],
        "mu_ice": mu_ice,
        "T_surf": prior_inputs["T_surf_K"],
        "D_H2O": D_H2O,
        "Q_v": Q_v,
        "Q_b": Q_b,
        "H_rad": H_rad,
        "P_tidal": P_tidal,
        "f_porosity": prior_inputs["f_porosity"],
        "f_salt": _DEFAULT_F_SALT,
        "T_phi": _DEFAULT_T_PHI,
        "B_k": _DEFAULT_B_K,
    }

    diagnostics = {
        "q_basal_target_mW_m2": prior_inputs["q_basal_target_mW_m2"],
        "q_radiogenic_mW_m2": q_radiogenic * 1.0e3,
        "q_silicate_tidal_mW_m2": q_silicate_tidal * 1.0e3,
        "q_basal_effective_mW_m2": (q_radiogenic + q_silicate_tidal) * 1.0e3,
        "enhancement_factor": scenario.enhancement_factor,
    }
    return SobolMappedSample(prior_inputs=prior_inputs, solver_params=solver_params, diagnostics=diagnostics)


def generate_sobol_design(
    problem: Mapping[str, Any],
    base_sample_size: int,
    *,
    calc_second_order: bool = False,
    scramble: bool = True,
    seed: Optional[int] = None,
) -> np.ndarray:
    from SALib.sample import sobol as salib_sobol

    if not is_power_of_two(base_sample_size):
        raise ValueError(
            f"Base Sobol sample size must be a power of two, received {base_sample_size}"
        )
    return salib_sobol.sample(
        problem,
        base_sample_size,
        calc_second_order=calc_second_order,
        scramble=scramble,
        seed=seed,
    )


def _evaluate_fixed_params(
    solver_params: Mapping[str, float],
    config: Any,
    *,
    physical_output_policy: str,
) -> Dict[str, float]:
    """
    Evaluate the thermal model on one deterministic parameter set.

    This mirrors the Monte Carlo worker logic closely, but retains explicit
    validity flags instead of silently dropping outputs.
    """
    from Boundary_Conditions import FixedTemperature
    from Convection import IceConvection  # noqa: F401 - mirrors MC worker import
    from Solver import Thermal_Solver
    from constants import Convection as ConvectionConstants
    from constants import Thermal

    T_surf = float(solver_params["T_surf"])
    D_H2O = float(solver_params["D_H2O"])
    H_rad = float(solver_params["H_rad"])
    P_tidal = float(solver_params["P_tidal"])

    R_rock = Planetary.RADIUS - D_H2O
    M_rock = (4.0 / 3.0) * np.pi * (R_rock ** 3) * _RHO_ROCK
    q_radiogenic = (H_rad * M_rock) / Planetary.AREA
    q_silicate_tidal = P_tidal / Planetary.AREA
    q_basal = q_radiogenic + q_silicate_tidal

    if physical_output_policy not in {"keep", "nan"}:
        raise ValueError(
            f"Unsupported physical_output_policy '{physical_output_policy}'. "
            "Use 'keep' or 'nan'."
        )

    if getattr(config, "use_warm_start", True) and q_basal > 0.0:
        delta_T = Thermal.MELT_TEMP - T_surf
        k_mean = Thermal.conductivity((T_surf + Thermal.MELT_TEMP) / 2.0)
        H_guess = (k_mean * delta_T) / q_basal
        if getattr(config, "use_convection", True):
            # Match the Monte_Carlo warm-start heuristic exactly so the Sobol
            # runs explore the same solver branch as the production MC engine.
            H_guess *= _WARM_START_CONVECTION_FACTOR
        H_guess = float(np.clip(H_guess, 5.0e3, 100.0e3))
    else:
        H_guess = float(config.initial_thickness)

    # Apply physics-selector overrides from config (if present).
    # Mirrors the 2D MC worker pattern (monte_carlo_2d.py lines 129-138).
    if hasattr(config, 'nu_scaling'):
        ConvectionConstants.NU_SCALING = config.nu_scaling
    if hasattr(config, 'conductivity_model'):
        from ConfigManager import ConfigManager
        _cfg = ConfigManager()
        _cfg._config.setdefault("thermal", {})["CONDUCTIVITY_MODEL"] = config.conductivity_model
    if hasattr(config, 'creep_model'):
        from ConfigManager import ConfigManager
        _cfg = ConfigManager()
        _cfg._config.setdefault("rheology", {})["CREEP_MODEL"] = config.creep_model
    if hasattr(config, 'grain_mode'):
        from ConfigManager import ConfigManager
        _cfg = ConfigManager()
        _cfg._config.setdefault("rheology", {})["GRAIN_MODE"] = config.grain_mode

    surface_bc = FixedTemperature(temperature=T_surf)
    solver = Thermal_Solver(
        nx=config.nx,
        thickness=H_guess,
        dt=config.dt,
        total_time=config.total_time,
        coordinate_system=config.coordinate_system,
        surface_bc=surface_bc,
        rannacher_steps=config.rannacher_steps,
        use_convection=config.use_convection,
        physics_params=dict(solver_params),
    )

    for _ in range(config.max_steps):
        velocity = solver.solve_step(q_basal)
        if abs(velocity) < config.eq_threshold:
            break

    H_km = float(solver.H / 1000.0)
    D_H2O_km = float(D_H2O / 1000.0)

    subcritical = False
    if config.use_convection and solver.convection_state is not None:
        state = solver.convection_state
        if state.D_conv > 0 and state.Ra < ConvectionConstants.RA_CRIT:
            subcritical = True

    physical_flag = 1.0
    if H_km <= 0.5 or H_km >= D_H2O_km * 0.99 or H_km > 200.0:
        physical_flag = 0.0
    if getattr(config, "reject_subcritical", False) and subcritical:
        physical_flag = 0.0

    if subcritical or solver.convection_state is None:
        D_cond_km = H_km
        D_conv_km = 0.0
        lid_fraction = 1.0
        Ra = float(solver.convection_state.Ra) if solver.convection_state else 0.0
        Nu = 1.0
        convective_flag = 0.0
    else:
        state = solver.convection_state
        D_cond_km = float(state.D_cond / 1000.0)
        D_conv_km = float(state.D_conv / 1000.0)
        lid_fraction = float(state.D_cond / solver.H) if solver.H > 0 else 1.0
        Ra = float(state.Ra)
        Nu = float(state.Nu)
        convective_flag = 1.0 if state.Ra >= ConvectionConstants.RA_CRIT else 0.0

    numerical_success = 1.0
    metrics = {
        "numerical_success": numerical_success,
        "physical_flag": physical_flag,
        "valid_flag": numerical_success * physical_flag,
        "subcritical_flag": 1.0 if subcritical else 0.0,
        "convective_flag": convective_flag,
        "thickness_km": H_km,
        "D_cond_km": D_cond_km,
        "D_conv_km": D_conv_km,
        "lid_fraction": lid_fraction,
        "Ra": Ra,
        "Nu": Nu,
        "q_radiogenic_mW_m2": q_radiogenic * 1.0e3,
        "q_silicate_tidal_mW_m2": q_silicate_tidal * 1.0e3,
        "q_basal_effective_mW_m2": q_basal * 1.0e3,
    }

    if physical_output_policy == "nan" and physical_flag == 0.0:
        for key in ("thickness_km", "D_cond_km", "D_conv_km", "lid_fraction", "Ra", "Nu"):
            metrics[key] = np.nan

    return metrics


def _sobol_worker(task: Any) -> Dict[str, Any]:
    sample_index, unit_sample, scenario_name, config, physical_output_policy = task
    mapped = map_unit_sample_to_model(unit_sample, scenario_name)
    try:
        metrics = _evaluate_fixed_params(
            mapped.solver_params,
            config,
            physical_output_policy=physical_output_policy,
        )
        error_type = ""
        error_message = ""
    except BaseException as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        metrics = {
            "numerical_success": 0.0,
            "physical_flag": 0.0,
            "valid_flag": 0.0,
            "subcritical_flag": np.nan,
            "convective_flag": np.nan,
            "thickness_km": np.nan,
            "D_cond_km": np.nan,
            "D_conv_km": np.nan,
            "lid_fraction": np.nan,
            "Ra": np.nan,
            "Nu": np.nan,
            "q_radiogenic_mW_m2": mapped.diagnostics["q_radiogenic_mW_m2"],
            "q_silicate_tidal_mW_m2": mapped.diagnostics["q_silicate_tidal_mW_m2"],
            "q_basal_effective_mW_m2": mapped.diagnostics["q_basal_effective_mW_m2"],
        }
        error_type = exc.__class__.__name__
        error_message = str(exc)

    return {
        "sample_index": sample_index,
        "prior_inputs": mapped.prior_inputs,
        "diagnostics": mapped.diagnostics,
        "metrics": metrics,
        "error_type": error_type,
        "error_message": error_message,
    }


def _empty_output_arrays(results: Sequence[Mapping[str, Any]], key: str) -> Dict[str, np.ndarray]:
    names = list(results[0][key].keys())
    return {name: np.empty(len(results), dtype=float) for name in names}


def evaluate_sobol_design(
    unit_design: np.ndarray,
    scenario_name: str,
    config: Any,
    *,
    n_workers: Optional[int] = None,
    physical_output_policy: str = "keep",
    verbose: bool = True,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Evaluate every row in a Sobol design and preserve design order.
    """
    n_rows = int(unit_design.shape[0])
    n_workers = n_workers or max(1, mp.cpu_count() - 1)
    tasks = [
        (i, unit_design[i], scenario_name, config, physical_output_policy)
        for i in range(n_rows)
    ]

    if n_workers > 1:
        chunksize = max(1, n_rows // (n_workers * 4))
        with mp.Pool(n_workers) as pool:
            iterator = pool.imap(_sobol_worker, tasks, chunksize=chunksize)
            results = []
            for i, result in enumerate(iterator, start=1):
                results.append(result)
                if verbose and i % max(1, n_rows // 10) == 0:
                    print(f"  Sobol progress: {100.0 * i / n_rows:5.1f}% ({i}/{n_rows})")
    else:
        results = []
        for i, task in enumerate(tasks, start=1):
            results.append(_sobol_worker(task))
            if verbose and i % max(1, n_rows // 10) == 0:
                print(f"  Sobol progress: {100.0 * i / n_rows:5.1f}% ({i}/{n_rows})")

    if not results:
        raise RuntimeError("Sobol evaluation produced no results")

    prior_inputs = _empty_output_arrays(results, "prior_inputs")
    diagnostics = _empty_output_arrays(results, "diagnostics")
    outputs = _empty_output_arrays(results, "metrics")
    error_types = np.empty(len(results), dtype="<U64")
    error_messages = np.empty(len(results), dtype="<U512")

    for row_index, result in enumerate(results):
        for key, value in result["prior_inputs"].items():
            prior_inputs[key][row_index] = value
        for key, value in result["diagnostics"].items():
            diagnostics[key][row_index] = value
        for key, value in result["metrics"].items():
            outputs[key][row_index] = value
        error_types[row_index] = result["error_type"]
        error_messages[row_index] = result["error_message"]

    return {
        "prior_inputs": prior_inputs,
        "diagnostics": diagnostics,
        "outputs": outputs,
        "errors": {
            "error_type": error_types,
            "error_message": error_messages,
        },
    }


def default_convergence_schedule(base_sample_size: int) -> List[int]:
    if not is_power_of_two(base_sample_size):
        raise ValueError("Sobol convergence schedule requires a power-of-two base sample size")
    min_power = max(7, int(np.log2(base_sample_size)) - 2)
    max_power = int(np.log2(base_sample_size))
    return [2 ** power for power in range(min_power, max_power + 1)]


def compute_sobol_indices(
    problem: Mapping[str, Any],
    outputs: Mapping[str, np.ndarray],
    *,
    output_names: Optional[Iterable[str]] = None,
    base_sample_sizes: Optional[Sequence[int]] = None,
    calc_second_order: bool = False,
    num_resamples: int = 1000,
    conf_level: float = 0.95,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    from SALib.analyze import sobol as salib_sobol

    available_outputs = list(outputs.keys())
    requested_outputs = list(output_names) if output_names is not None else list(DEFAULT_ANALYSIS_OUTPUTS)
    requested_outputs = [name for name in requested_outputs if name in available_outputs]

    if not requested_outputs:
        raise ValueError("No valid Sobol outputs were requested")

    if base_sample_sizes is None:
        full_rows = len(next(iter(outputs.values())))
        d_eff = effective_dimension(problem)
        divisor = (2 * d_eff + 2) if calc_second_order else (d_eff + 2)
        max_base = full_rows // divisor
        base_sample_sizes = default_convergence_schedule(max_base)

    factor_labels = ordered_unique(problem["groups"]) if "groups" in problem else list(problem["names"])

    results: Dict[str, Any] = {}
    for output_name in requested_outputs:
        output_result: Dict[str, Any] = {"convergence": []}
        y_full = np.asarray(outputs[output_name], dtype=float)

        for base_n in base_sample_sizes:
            n_rows = expected_sobol_rows(problem, base_n, calc_second_order)
            y = y_full[:n_rows]
            if not np.all(np.isfinite(y)):
                output_result["skip_reason"] = (
                    f"Output '{output_name}' contains non-finite values in the first {n_rows} rows"
                )
                break

            try:
                Si = salib_sobol.analyze(
                    problem,
                    y,
                    calc_second_order=calc_second_order,
                    num_resamples=num_resamples,
                    conf_level=conf_level,
                    print_to_console=False,
                    seed=seed,
                )
                output_result["convergence"].append({"N": int(base_n), "Si": Si})
            except (ValueError, TypeError) as exc:
                import logging
                logging.getLogger(__name__).warning(
                    "SALib analyze failed for output '%s' at N=%d: %s",
                    output_name, base_n, exc,
                )
                output_result.setdefault("skip_reason",
                    f"SALib error for '{output_name}' at N={base_n}: {exc}"
                )

        if output_result["convergence"]:
            output_result["final"] = output_result["convergence"][-1]
            output_result["factor_labels"] = factor_labels
        results[output_name] = output_result

    return results


def sobol_results_to_rows(
    problem: Mapping[str, Any],
    sobol_results: Mapping[str, Any],
    *,
    include_second_order: bool = False,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    factor_labels = ordered_unique(problem["groups"]) if "groups" in problem else list(problem["names"])

    for output_name, output_result in sobol_results.items():
        if "convergence" not in output_result or not output_result["convergence"]:
            rows.append(
                {
                    "output": output_name,
                    "sample_size": "",
                    "index_type": "",
                    "factor": "",
                    "S1": "",
                    "S1_conf": "",
                    "ST": "",
                    "ST_conf": "",
                    "S2": "",
                    "S2_conf": "",
                    "skip_reason": output_result.get("skip_reason", "No Sobol result"),
                }
            )
            continue

        for convergence_item in output_result["convergence"]:
            N = convergence_item["N"]
            Si = convergence_item["Si"]
            for i, factor in enumerate(factor_labels):
                rows.append(
                    {
                        "output": output_name,
                        "sample_size": N,
                        "index_type": "main",
                        "factor": factor,
                        "S1": float(Si["S1"][i]),
                        "S1_conf": float(Si["S1_conf"][i]),
                        "ST": float(Si["ST"][i]),
                        "ST_conf": float(Si["ST_conf"][i]),
                        "S2": "",
                        "S2_conf": "",
                        "skip_reason": "",
                    }
                )

            if include_second_order and "S2" in Si:
                for i, factor_i in enumerate(factor_labels):
                    for j in range(i + 1, len(factor_labels)):
                        rows.append(
                            {
                                "output": output_name,
                                "sample_size": N,
                                "index_type": "interaction",
                                "factor": f"{factor_i} x {factor_labels[j]}",
                                "S1": "",
                                "S1_conf": "",
                                "ST": "",
                                "ST_conf": "",
                                "S2": float(Si["S2"][i, j]),
                                "S2_conf": float(Si["S2_conf"][i, j]),
                                "skip_reason": "",
                            }
                        )
    return rows


def summarize_top_total_indices(
    problem: Mapping[str, Any],
    sobol_results: Mapping[str, Any],
    *,
    top_n: int = 3,
) -> Dict[str, List[Dict[str, float]]]:
    factor_labels = ordered_unique(problem["groups"]) if "groups" in problem else list(problem["names"])
    summary: Dict[str, List[Dict[str, float]]] = {}
    for output_name, output_result in sobol_results.items():
        final = output_result.get("final")
        if not final:
            continue
        Si = final["Si"]
        st = np.asarray(Si["ST"], dtype=float)
        order = np.argsort(st)[::-1][:top_n]
        summary[output_name] = [
            {
                "factor": factor_labels[i],
                "ST": float(st[i]),
                "ST_conf": float(Si["ST_conf"][i]),
                "S1": float(Si["S1"][i]),
                "S1_conf": float(Si["S1_conf"][i]),
            }
            for i in order
        ]
    return summary
