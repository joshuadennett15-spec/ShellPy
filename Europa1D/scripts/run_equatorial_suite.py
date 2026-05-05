"""
Equatorial-proxy Monte Carlo suite with three ocean heat transport modes.

Runs Andrade rheology with audited priors + equatorial overrides.
Produces one NPZ per mode for downstream Juno comparison.

Shared seed: all three modes use the same base seed so that the
sampler-side parameter draws (d_grain, Q_v, Q_b, etc.) are drawn
from the same RNG sequence. This reduces sampler-side noise in
mode-to-mode comparisons. Note: the saved NPZ arrays are NOT
index-aligned across modes because MonteCarloRunner uses
imap_unordered and filters invalid draws. For true paired
analysis, match draws by sample ID; for aggregate statistics
(CDFs, Bayes factors), the shared seed is sufficient.

reject_subcritical=False matches the audited global Andrade baseline
(run_andrade_15k.py) so that the comparison isolates equatorial
forcing, not branch-handling differences.

Modes:
  depleted_strong (0.55x) — strong equatorial depletion (Lemasquerier 2023 tidal-dominant)
  depleted        (0.67x) — conservative equatorial depletion (Lemasquerier 2023 balanced)
  baseline        (1.0x)  — uniform ocean heat transport (Ashkenazy & Tziperman 2021)
  moderate        (1.2x)  — Soderlund (2014) equatorial enhancement proxy
  strong          (1.5x)  — upper-bound equatorial enhancement

Config must be set to Andrade in src/config.json.
"""
import argparse
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import multiprocessing as mp
from runtime_support import configure_numeric_runtime, default_worker_count, resolve_worker_count

configure_numeric_runtime()

from Monte_Carlo import MonteCarloRunner, SolverConfig, save_results
from audited_equatorial_sampler import AuditedEquatorialSampler
from constants import Rheology

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')

SEED = 10042
N_ITERATIONS = 10000

MODES = [
    ("depleted_strong", 0.55),
    ("depleted",        0.67),
    ("baseline",        1.0),
    ("moderate",        1.2),
    ("strong",          1.5),
]


# Module-level sampler classes (picklable on Windows spawn-based mp)
class DepletedStrongSampler(AuditedEquatorialSampler):
    """Lemasquerier (2023) tidal-dominant: 0.273/0.50 = 0.55x equatorial."""
    def __init__(self, **kwargs):
        super().__init__(enhancement_factor=0.55, **kwargs)


class DepletedSampler(AuditedEquatorialSampler):
    """Lemasquerier (2023) balanced: 0.333/0.50 = 0.67x equatorial."""
    def __init__(self, **kwargs):
        super().__init__(enhancement_factor=0.67, **kwargs)


class BaselineSampler(AuditedEquatorialSampler):
    def __init__(self, **kwargs):
        super().__init__(enhancement_factor=1.0, **kwargs)


class ModerateSampler(AuditedEquatorialSampler):
    def __init__(self, **kwargs):
        super().__init__(enhancement_factor=1.2, **kwargs)


class StrongSampler(AuditedEquatorialSampler):
    def __init__(self, **kwargs):
        super().__init__(enhancement_factor=1.5, **kwargs)


_SAMPLER_MAP = {
    "depleted_strong": DepletedStrongSampler,
    "depleted":        DepletedSampler,
    "baseline":        BaselineSampler,
    "moderate":        ModerateSampler,
    "strong":          StrongSampler,
}


def _default_workers() -> int:
    """Use the shared Windows-safe worker default for 1D Monte Carlo runs."""
    return default_worker_count()


def _output_path(label: str) -> str:
    """Canonical output path for a given mode label."""
    return os.path.join(RESULTS_DIR, f"eq_{label}_andrade.npz")


def _is_valid_result(path: str, expected_n: int) -> bool:
    """Check if an existing .npz result file is valid and complete."""
    if not os.path.exists(path):
        return False
    try:
        data = np.load(path)
        n_valid = int(data["n_valid"]) if "n_valid" in data else 0
        has_keys = all(k in data for k in ("thicknesses_km", "D_cond_km", "Ra_values"))
        return has_keys and n_valid >= expected_n * 0.95  # allow tiny rejection margin
    except Exception:
        return False


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--iterations",
        type=int,
        default=N_ITERATIONS,
        help=f"Monte Carlo iterations per mode (default: {N_ITERATIONS}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help=f"Shared random seed for all modes (default: {SEED}).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=_default_workers(),
        help="Parallel worker processes. Windows default is capped conservatively.",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=list(_SAMPLER_MAP.keys()),
        default=None,
        help="Optional subset of equatorial modes to run.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Skip modes whose output .npz already exists and is valid (default: on).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run all modes even if results exist (overrides --resume).",
    )
    return parser.parse_args()


def run_mode(label, enhancement_factor, iterations: int, seed: int, workers: int):
    print(f"\n{'=' * 60}")
    print(f"EQUATORIAL MODE: {label} ({enhancement_factor:.1f}x tidal)")
    print(f"  N = {iterations:,}, seed = {seed} (paired), workers = {workers}")
    print(f"{'=' * 60}")

    config = SolverConfig(reject_subcritical=False)

    runner = MonteCarloRunner(
        n_iterations=iterations,
        seed=seed,
        verbose=True,
        n_workers=workers,
        config=config,
        sampler_class=_SAMPLER_MAP[label],
    )
    results = runner.run()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_path = os.path.join(RESULTS_DIR, f"eq_{label}_andrade.npz")
    save_results(results, output_path)

    print(f"\n--- {label} ({enhancement_factor:.1f}x) RESULTS ---")
    print(f"  CBE:     {results.cbe_km:.1f} km")
    print(f"  Median:  {results.median_km:.1f} km")
    print(f"  1-sigma: [{results.sigma_1_low_km:.1f}, {results.sigma_1_high_km:.1f}] km")
    print(f"  Valid:   {results.n_valid}/{results.n_iterations}")

    return output_path


def main():
    args = _parse_args()
    workers = resolve_worker_count(args.workers)
    print(f"Rheology model: {Rheology.MODEL}")
    assert Rheology.MODEL == "Andrade", f"Expected Andrade, got {Rheology.MODEL}"

    resume = args.resume and not args.force

    paths = {}
    skipped = []
    modes = [item for item in MODES if args.modes is None or item[0] in args.modes]

    print(f"\nModes: {len(modes)} | Resume: {resume}")
    for label, factor in modes:
        out = _output_path(label)
        if resume and _is_valid_result(out, args.iterations):
            data = np.load(out)
            n_valid = int(data["n_valid"])
            print(f"\n  SKIP {label} ({factor:.2f}x) — valid result exists "
                  f"({n_valid:,} samples): {out}")
            paths[label] = out
            skipped.append(label)
            continue

        paths[label] = run_mode(
            label,
            factor,
            iterations=args.iterations,
            seed=args.seed,
            workers=workers,
        )

    print(f"\n{'=' * 60}")
    print("EQUATORIAL SUITE COMPLETE")
    if skipped:
        print(f"  Skipped (resumed): {', '.join(skipped)}")
    for label, path in paths.items():
        status = "resumed" if label in skipped else "computed"
        print(f"  {label} [{status}]: {path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
