"""
Full 2D Monte Carlo runs for literature-backed scenarios.

Supports pause/resume via checkpoint files. Each scenario's iterations are
split into batches. After each batch the raw per-sample results are saved to
a checkpoint file. On --resume, completed batches are skipped.

Ctrl+C between batches saves progress cleanly.
"""
import argparse
import multiprocessing as mp
import os
import signal
import sys
import time

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.join(_SCRIPT_DIR, "..")
sys.path.insert(0, _PROJECT_DIR)
sys.path.insert(0, os.path.join(_PROJECT_DIR, "src"))
import src  # noqa: F401

import numpy as np

from literature_scenarios import DEFAULT_SCENARIO, get_scenario, list_scenarios
from monte_carlo_2d import MonteCarloRunner2D, save_results_2d

# ── Checkpoint helpers ───────────────────────────────────────────────────────

_SAMPLE_FIELDS = ("H_km", "D_cond_km", "D_conv_km", "Ra", "Nu",
                   "lid_fraction", "T_c", "Ti")

_interrupted = False


def _checkpoint_path(results_dir: str, scenario_name: str) -> str:
    return os.path.join(results_dir, f"mc_2d_{scenario_name}_checkpoint.npz")


def _load_checkpoint(path: str):
    """Load checkpoint. Returns (list_of_sample_dicts, n_completed)."""
    if not os.path.exists(path):
        return [], 0
    d = np.load(path, allow_pickle=True)
    n = int(d["n_completed"])
    samples = []
    for i in range(n):
        sample = {}
        for field in _SAMPLE_FIELDS:
            key = f"s{i}_{field}"
            if key in d:
                sample[field] = d[key]
        if sample:
            samples.append(sample)
        else:
            samples.append(None)
    return samples, n


def _save_checkpoint(path: str, samples: list, n_completed: int):
    """Save checkpoint with all raw per-sample arrays."""
    save_dict = {"n_completed": n_completed}
    for i, sample in enumerate(samples):
        if sample is not None:
            for field in _SAMPLE_FIELDS:
                if field in sample:
                    save_dict[f"s{i}_{field}"] = sample[field]
    np.savez(path, **save_dict)


def _delete_checkpoint(path: str):
    if os.path.exists(path):
        os.remove(path)


# ── Batch runner ─────────────────────────────────────────────────────────────

def run_mc_scenario_batched(
    scenario_name: str,
    results_dir: str,
    iterations: int,
    seed: int,
    n_workers: int,
    n_lat: int,
    nx: int,
    dt: float,
    max_steps: int,
    grain_latitude_mode: str = "global",
    q_tidal_scale: float = 1.0,
    batch_size: int = 50,
    resume: bool = False,
) -> str:
    """Run one MC scenario in resumable batches."""
    scenario = get_scenario(scenario_name)
    ckpt_path = _checkpoint_path(results_dir, scenario_name)
    os.makedirs(results_dir, exist_ok=True)

    # Load existing progress
    if resume:
        samples, n_done = _load_checkpoint(ckpt_path)
    else:
        samples, n_done = [], 0
        _delete_checkpoint(ckpt_path)

    if n_done >= iterations:
        print(f"\n=== {scenario.name}: already complete ({n_done}/{iterations}) ===")
    else:
        print(f"\n=== {scenario.name}: {scenario.citation} ===")
        print(f"  {scenario.description}")
        if grain_latitude_mode != "global":
            print(f"  grain_latitude_mode: {grain_latitude_mode}")
        if n_done > 0:
            print(f"  Resuming from {n_done}/{iterations} completed")

    # Process remaining iterations in batches
    while n_done < iterations and not _interrupted:
        batch_start = n_done
        batch_count = min(batch_size, iterations - n_done)
        batch_seed = seed + batch_start

        print(f"\n  Batch [{batch_start+1}..{batch_start+batch_count}] / {iterations}  "
              f"(seed offset +{batch_start})")
        t0 = time.time()

        runner = MonteCarloRunner2D(
            n_iterations=batch_count,
            seed=batch_seed,
            n_workers=n_workers,
            n_lat=n_lat,
            nx=nx,
            dt=dt,
            use_convection=True,
            max_steps=max_steps,
            ocean_pattern=scenario.ocean_pattern,
            q_star=scenario.q_star if scenario.q_star > 0 else None,
            grain_latitude_mode=grain_latitude_mode,
            q_tidal_scale=q_tidal_scale,
            verbose=False,
        )
        batch_results = runner.run()

        # Unpack the MonteCarloResults2D into per-sample dicts
        for i in range(batch_results.n_valid):
            samples.append({
                "H_km": batch_results.H_profiles[i],
                "D_cond_km": batch_results.D_cond_profiles[i] if batch_results.D_cond_profiles is not None else np.full(n_lat, np.nan),
                "D_conv_km": batch_results.D_conv_profiles[i] if batch_results.D_conv_profiles is not None else np.full(n_lat, np.nan),
                "Ra": batch_results.Ra_profiles[i] if batch_results.Ra_profiles is not None else np.full(n_lat, np.nan),
                "Nu": batch_results.Nu_profiles[i] if batch_results.Nu_profiles is not None else np.full(n_lat, np.nan),
                "lid_fraction": batch_results.lid_fraction_profiles[i] if batch_results.lid_fraction_profiles is not None else np.full(n_lat, np.nan),
                "T_c": batch_results.T_c_profiles[i] if batch_results.T_c_profiles is not None else np.full(n_lat, np.nan),
                "Ti": batch_results.Ti_profiles[i] if batch_results.Ti_profiles is not None else np.full(n_lat, np.nan),
            })
        # Also count None results (failed samples) to keep seed alignment
        n_failed = batch_count - batch_results.n_valid
        for _ in range(n_failed):
            samples.append(None)

        n_done += batch_count
        elapsed = time.time() - t0

        print(f"    {batch_results.n_valid}/{batch_count} valid, {elapsed:.1f}s")

        # Checkpoint after each batch
        _save_checkpoint(ckpt_path, samples, n_done)
        print(f"    Checkpoint saved ({n_done}/{iterations})")

    # If interrupted before completion, leave checkpoint for --resume
    if n_done < iterations:
        valid_so_far = sum(1 for s in samples if s is not None)
        print(f"\n  Paused: {n_done}/{iterations} iterations done "
              f"({valid_so_far} valid). Use --resume to continue.")
        return ckpt_path

    # Final aggregation: rebuild a MonteCarloResults2D from all samples
    # and save the production output
    valid_samples = [s for s in samples if s is not None]
    n_valid = len(valid_samples)
    print(f"\n  Final: {n_valid}/{iterations} valid samples")

    # Rebuild a runner just for aggregation metadata
    runner = MonteCarloRunner2D(
        n_iterations=iterations,
        seed=seed,
        n_workers=1,
        n_lat=n_lat,
        nx=nx,
        dt=dt,
        use_convection=True,
        max_steps=max_steps,
        ocean_pattern=scenario.ocean_pattern,
        q_star=scenario.q_star if scenario.q_star > 0 else None,
        grain_latitude_mode=grain_latitude_mode,
        q_tidal_scale=q_tidal_scale,
    )

    # Stack arrays and build the results object via the runner's aggregation
    # We call runner._aggregate (if available) or build manually
    from monte_carlo_2d import MonteCarloResults2D
    from profile_diagnostics import band_mean_samples, LOW_LAT_BAND, HIGH_LAT_BAND

    H_profiles = np.array([s["H_km"] for s in valid_samples])
    D_cond = np.array([s["D_cond_km"] for s in valid_samples])
    D_conv = np.array([s["D_conv_km"] for s in valid_samples])
    Ra = np.array([s["Ra"] for s in valid_samples])
    Nu = np.array([s["Nu"] for s in valid_samples])
    lid_frac = np.array([s["lid_fraction"] for s in valid_samples])
    T_c_stack = np.array([s["T_c"] for s in valid_samples])
    Ti_stack = np.array([s["Ti"] for s in valid_samples])
    latitudes_deg = np.linspace(0, 90, n_lat)

    conv_fraction_stack = np.where(H_profiles > 0, D_conv / H_profiles, 0.0)

    results = MonteCarloResults2D(
        H_profiles=H_profiles,
        latitudes_deg=latitudes_deg,
        n_iterations=iterations,
        n_valid=n_valid,
        H_median=np.median(H_profiles, axis=0),
        H_mean=np.mean(H_profiles, axis=0),
        H_sigma_low=np.percentile(H_profiles, 15.87, axis=0),
        H_sigma_high=np.percentile(H_profiles, 84.13, axis=0),
        runtime_seconds=0.0,
        ocean_pattern=scenario.ocean_pattern,
        ocean_amplitude=getattr(runner, "ocean_amplitude", 0.0),
        T_floor=46.0,
        q_star=scenario.q_star,
        q_tidal_scale=q_tidal_scale,
        D_cond_profiles=D_cond,
        D_conv_profiles=D_conv,
        Ra_profiles=Ra,
        Nu_profiles=Nu,
        lid_fraction_profiles=lid_frac,
        T_c_profiles=T_c_stack,
        Ti_profiles=Ti_stack,
        D_cond_median=np.median(D_cond, axis=0),
        D_cond_mean=np.mean(D_cond, axis=0),
        D_cond_sigma_low=np.percentile(D_cond, 15.87, axis=0),
        D_cond_sigma_high=np.percentile(D_cond, 84.13, axis=0),
        D_conv_median=np.median(D_conv, axis=0),
        D_conv_mean=np.mean(D_conv, axis=0),
        D_conv_sigma_low=np.percentile(D_conv, 15.87, axis=0),
        D_conv_sigma_high=np.percentile(D_conv, 84.13, axis=0),
        T_c_median=np.median(T_c_stack, axis=0),
        T_c_mean=np.mean(T_c_stack, axis=0),
        Ti_median=np.median(Ti_stack, axis=0),
        Ti_mean=np.mean(Ti_stack, axis=0),
        conv_fraction_median=np.median(conv_fraction_stack, axis=0),
        conv_fraction_mean=np.mean(conv_fraction_stack, axis=0),
        conv_fraction_sigma_low=np.percentile(conv_fraction_stack, 15.87, axis=0),
        conv_fraction_sigma_high=np.percentile(conv_fraction_stack, 84.13, axis=0),
        H_low_band=band_mean_samples(latitudes_deg, H_profiles, LOW_LAT_BAND),
        H_high_band=band_mean_samples(latitudes_deg, H_profiles, HIGH_LAT_BAND),
        D_cond_low_band=band_mean_samples(latitudes_deg, D_cond, LOW_LAT_BAND),
        D_cond_high_band=band_mean_samples(latitudes_deg, D_cond, HIGH_LAT_BAND),
    )

    output_path = os.path.join(results_dir, f"mc_2d_{scenario.name}_{iterations}.npz")
    save_results_2d(results, output_path)

    # Clean up checkpoint now that production file is written
    _delete_checkpoint(ckpt_path)
    print(f"  Checkpoint cleaned up")

    return output_path


# ── CLI ──────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run 2D Monte Carlo for literature scenarios (pausable)."
    )
    parser.add_argument(
        "--scenario",
        choices=["all", *list_scenarios()],
        default=DEFAULT_SCENARIO,
        help="Scenario preset to run. Use 'all' to loop over all literature presets.",
    )
    parser.add_argument("--iterations", type=int, default=1000, help="Number of Monte Carlo iterations.")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
    parser.add_argument("--n-lat", type=int, default=37, help="Number of latitude columns.")
    parser.add_argument("--nx", type=int, default=31, help="Radial nodes per column.")
    parser.add_argument("--dt", type=float, default=1e12, help="Time step in seconds.")
    parser.add_argument("--max-steps", type=int, default=1500, help="Maximum solver steps per sample.")
    parser.add_argument(
        "--n-workers",
        type=int,
        default=max(1, mp.cpu_count() - 1),
        help="Number of worker processes.",
    )
    parser.add_argument("--q-tidal-scale", type=float, default=1.20, help="Scale factor applied to ocean heat flux.")
    parser.add_argument(
        "--grain-mode",
        choices=["global", "strain"],
        default="global",
        help="Grain latitude mode: 'global' (benchmark) or 'strain' (recrystallization).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=50,
        help="Samples per checkpoint batch (default 50).",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from checkpoint if one exists.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    mp.freeze_support()
    args = _parse_args()

    results_dir = os.path.join(_PROJECT_DIR, "results")
    scenario_names = list_scenarios() if args.scenario == "all" else (args.scenario,)

    # Graceful Ctrl+C: let the current batch finish, then stop
    def _handle_sigint(sig, frame):
        global _interrupted
        if _interrupted:
            print("\n  Force quit (second Ctrl+C)")
            sys.exit(1)
        _interrupted = True
        print("\n  Ctrl+C received — finishing current batch, then stopping.")
        print("  (Press Ctrl+C again to force quit)")

    signal.signal(signal.SIGINT, _handle_sigint)

    saved_paths = []
    for offset, scenario_name in enumerate(scenario_names):
        if _interrupted:
            print(f"\n  Skipping {scenario_name} (interrupted)")
            continue

        output_path = run_mc_scenario_batched(
            scenario_name=scenario_name,
            results_dir=results_dir,
            iterations=args.iterations,
            seed=args.seed + offset * 10000,
            n_workers=args.n_workers,
            n_lat=args.n_lat,
            nx=args.nx,
            dt=args.dt,
            max_steps=args.max_steps,
            grain_latitude_mode=args.grain_mode,
            q_tidal_scale=args.q_tidal_scale,
            batch_size=args.batch_size,
            resume=args.resume,
        )
        saved_paths.append(output_path)

    print("\nSaved MC outputs:")
    for path in saved_paths:
        print(f"  - {path}")
