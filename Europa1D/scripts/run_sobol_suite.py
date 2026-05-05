"""
Run a literature-grounded Sobol sensitivity study for the 1D ice-shell model.

This script generates a dedicated Sobol design, maps the unit-cube design to
the audited priors, evaluates the thermal model, and computes Sobol first- and
total-order indices with convergence checkpoints.
"""
import argparse
from collections import Counter
import csv
import json
import os
from pathlib import Path
import sys
from typing import Iterable, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from runtime_support import configure_numeric_runtime

configure_numeric_runtime()

import numpy as np

from Monte_Carlo import SolverConfig
from sobol_workflow import (
    DEFAULT_ANALYSIS_OUTPUTS,
    SCENARIOS,
    build_salib_problem,
    compute_sobol_indices,
    default_convergence_schedule,
    effective_dimension,
    evaluate_sobol_design,
    expected_sobol_rows,
    generate_sobol_design,
    get_sobol_scenario,
    is_power_of_two,
    ordered_unique,
    sobol_results_to_rows,
    summarize_top_total_indices,
)


RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "sobol"

LITERATURE = [
    {
        "title": "Sobol (2001) Global sensitivity indices for nonlinear mathematical models and their Monte Carlo estimates",
        "url": "https://doi.org/10.1016/S0378-4754(00)00270-6",
    },
    {
        "title": "Saltelli et al. (2010) Variance based sensitivity analysis of model output",
        "url": "https://doi.org/10.1016/j.cpc.2009.09.018",
    },
    {
        "title": "SALib documentation: sample.sobol and analyze.sobol",
        "url": "https://salib.readthedocs.io/en/latest/api/SALib.html",
    },
    {
        "title": "Saltelli et al. (2019) Why so many published sensitivity analyses are false",
        "url": "https://doi.org/10.1016/j.envsoft.2019.01.012",
    },
    {
        "title": "Sarrazin, Pianosi, Wagener (2016) Global Sensitivity Analysis of environmental models: Convergence and validation",
        "url": "https://doi.org/10.1016/j.envsoft.2016.02.005",
    },
    {
        "title": "Ryan et al. (2018) Fast sensitivity analysis methods for computationally expensive models",
        "url": "https://doi.org/10.5194/gmd-11-3131-2018",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario",
        default="global_audited",
        choices=sorted(SCENARIOS.keys()),
        help="Audited scenario to run.",
    )
    parser.add_argument(
        "--base-samples",
        type=int,
        default=512,
        help="Base Sobol sample size N. Use a power of two.",
    )
    parser.add_argument(
        "--schedule",
        default="",
        help="Comma-separated convergence checkpoints in base Sobol samples.",
    )
    parser.add_argument(
        "--grouped",
        action="store_true",
        help="Run grouped Sobol indices by physics block instead of per-parameter indices.",
    )
    parser.add_argument(
        "--second-order",
        action="store_true",
        help="Also compute second-order indices. This is much more expensive.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Sobol scrambling / bootstrap seed.",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=1,
        help="Number of worker processes for the model evaluations.",
    )
    parser.add_argument(
        "--num-resamples",
        type=int,
        default=1000,
        help="Bootstrap resamples for SALib confidence intervals.",
    )
    parser.add_argument(
        "--conf-level",
        type=float,
        default=0.95,
        help="Confidence level for Sobol index intervals.",
    )
    parser.add_argument(
        "--outputs",
        default=",".join(DEFAULT_ANALYSIS_OUTPUTS),
        help="Comma-separated list of outputs to analyze.",
    )
    parser.add_argument(
        "--physical-output-policy",
        default="keep",
        choices=("keep", "nan"),
        help="How to treat non-physical outputs before Sobol analysis.",
    )
    parser.add_argument(
        "--apply-physical-filters",
        action="store_const",
        const="nan",
        dest="physical_output_policy",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--reject-subcritical",
        action="store_true",
        help="Mirror the strict Monte Carlo filter for subcritical convective layers.",
    )
    parser.add_argument(
        "--run-label",
        default="",
        help="Optional custom output folder name.",
    )
    parser.add_argument(
        "--nx",
        type=int,
        default=31,
        help="Finite-difference grid size.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1500,
        help="Maximum solver steps per evaluation.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress updates during design evaluation.",
    )
    return parser.parse_args()


def parse_schedule(raw: str, base_samples: int) -> List[int]:
    if not raw.strip():
        return default_convergence_schedule(base_samples)
    schedule = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not schedule:
        raise ValueError("Schedule string produced no checkpoint sizes")
    for value in schedule:
        if value > base_samples:
            raise ValueError(f"Schedule value {value} exceeds base sample size {base_samples}")
        if not is_power_of_two(value):
            raise ValueError(f"Schedule value {value} is not a power of two")
    return sorted(set(schedule))


def parse_outputs(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def build_run_label(args: argparse.Namespace) -> str:
    if args.run_label:
        return args.run_label
    suffix = "groups" if args.grouped else "params"
    return f"{args.scenario}_sobol_{suffix}_N{args.base_samples}"


def write_csv(rows: Iterable[dict], path: Path) -> None:
    rows = list(rows)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_error_counts(error_types: np.ndarray, error_messages: np.ndarray) -> List[dict]:
    counts = Counter()
    for error_type, error_message in zip(error_types, error_messages):
        if not error_type:
            continue
        counts[(str(error_type), str(error_message))] += 1
    return [
        {"error_type": key[0], "error_message": key[1], "count": count}
        for key, count in counts.most_common()
    ]


def main() -> None:
    args = parse_args()

    if not is_power_of_two(args.base_samples):
        raise ValueError(f"--base-samples must be a power of two, received {args.base_samples}")

    outputs = parse_outputs(args.outputs)
    schedule = parse_schedule(args.schedule, args.base_samples)
    run_label = build_run_label(args)
    output_dir = RESULTS_DIR / run_label
    output_dir.mkdir(parents=True, exist_ok=True)

    scenario = get_sobol_scenario(args.scenario)
    problem = build_salib_problem(args.scenario, grouped=args.grouped)
    factor_labels = ordered_unique(problem["groups"]) if "groups" in problem else list(problem["names"])

    config = SolverConfig(reject_subcritical=args.reject_subcritical)
    config.nx = args.nx
    config.max_steps = args.max_steps

    print("=" * 72)
    print(f"SOBOL RUN: {scenario.label}")
    print(f"  Run label: {run_label}")
    print(f"  Grouped: {args.grouped}")
    print(f"  Base samples: {args.base_samples}")
    print(f"  Effective dimension: {effective_dimension(problem)}")
    print(f"  Expected evaluations: {expected_sobol_rows(problem, args.base_samples, args.second_order):,}")
    print(f"  Convergence schedule: {schedule}")
    print(f"  Outputs: {', '.join(outputs)}")
    print(f"  Non-physical output policy: {args.physical_output_policy}")
    print("=" * 72)
    if len(schedule) == 1:
        print("Warning: only one Sobol checkpoint is configured; convergence cannot be compared.")

    unit_design = generate_sobol_design(
        problem,
        args.base_samples,
        calc_second_order=args.second_order,
        seed=args.seed,
    )

    evaluation = evaluate_sobol_design(
        unit_design,
        args.scenario,
        config,
        n_workers=args.n_workers,
        physical_output_policy=args.physical_output_policy,
        verbose=not args.quiet,
    )

    sobol_results = compute_sobol_indices(
        problem,
        evaluation["outputs"],
        output_names=outputs,
        base_sample_sizes=schedule,
        calc_second_order=args.second_order,
        num_resamples=args.num_resamples,
        conf_level=args.conf_level,
        seed=args.seed,
    )

    summary = summarize_top_total_indices(problem, sobol_results)
    csv_rows = sobol_results_to_rows(
        problem,
        sobol_results,
        include_second_order=args.second_order,
    )

    np.savez_compressed(
        output_dir / f"{run_label}_design.npz",
        X_unit=unit_design,
        **{f"input_{name}": values for name, values in evaluation["prior_inputs"].items()},
        **{f"diag_{name}": values for name, values in evaluation["diagnostics"].items()},
        **{f"output_{name}": values for name, values in evaluation["outputs"].items()},
        error_type=evaluation["errors"]["error_type"],
        error_message=evaluation["errors"]["error_message"],
    )
    write_csv(csv_rows, output_dir / f"{run_label}_indices.csv")

    try:
        import SALib  # type: ignore

        salib_version = getattr(SALib, "__version__", "unknown")
    except Exception:
        salib_version = "unavailable"

    error_counts = summarize_error_counts(
        evaluation["errors"]["error_type"],
        evaluation["errors"]["error_message"],
    )

    manifest = {
        "run_label": run_label,
        "scenario": args.scenario,
        "scenario_label": scenario.label,
        "enhancement_factor": scenario.enhancement_factor,
        "grouped": args.grouped,
        "factor_labels": factor_labels,
        "parameter_names": list(problem["names"]),
        "base_samples": args.base_samples,
        "calc_second_order": args.second_order,
        "schedule": schedule,
        "seed": args.seed,
        "salib_version": salib_version,
        "n_workers": args.n_workers,
        "physical_output_policy": args.physical_output_policy,
        "reject_subcritical": args.reject_subcritical,
        "expected_evaluations": expected_sobol_rows(problem, args.base_samples, args.second_order),
        "numerical_success_rate": float(np.nanmean(evaluation["outputs"]["numerical_success"])),
        "n_numerical_failures": int(np.sum(evaluation["outputs"]["numerical_success"] == 0.0)),
        "physical_valid_rate": float(np.nanmean(evaluation["outputs"]["physical_flag"])),
        "error_counts": error_counts,
        "requested_outputs": outputs,
        "literature": LITERATURE,
        "top_total_order": summary,
    }
    with (output_dir / f"{run_label}_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print("\nTop total-order indices at final N")
    for output_name, top_rows in summary.items():
        print(f"\n{output_name}:")
        for row in top_rows:
            print(
                f"  {row['factor']}: "
                f"ST={row['ST']:.3f} +/- {row['ST_conf']:.3f}, "
                f"S1={row['S1']:.3f} +/- {row['S1_conf']:.3f}"
            )

    skipped = {
        output_name: result.get("skip_reason")
        for output_name, result in sobol_results.items()
        if result.get("skip_reason")
    }
    if skipped:
        print("\nSkipped outputs")
        for output_name, reason in skipped.items():
            print(f"  {output_name}: {reason}")

    if error_counts:
        print("\nNumerical failures")
        for error in error_counts[:5]:
            print(f"  {error['count']}x {error['error_type']}: {error['error_message']}")

    print("\nSaved files")
    print(f"  {output_dir / f'{run_label}_design.npz'}")
    print(f"  {output_dir / f'{run_label}_indices.csv'}")
    print(f"  {output_dir / f'{run_label}_manifest.json'}")


if __name__ == "__main__":
    main()
