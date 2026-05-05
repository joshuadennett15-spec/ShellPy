"""Compute equatorial D_conv threshold fractions for Discussion §5 ¶5.

For each 2D transport-scenario archive (500-member ensembles, 37 latitudes),
reports the fraction of equatorial ensemble members satisfying
D_conv >= 8 km and D_conv >= 15 km.

Thresholds from Nimmo & Manga (2002):
  8 km  — coherent warm-ice diapirs (diapir spacing >= 5–10 km)
  15 km — widespread chaos coverage (~40% of surface)

Archives live in Europa2D/results/ (2D latitude-aware MC).
Each NPZ has:
  latitudes_deg : (37,) float64  — 0 to 90 degrees (equator is index 0)
  D_conv_profiles : (500, 37) float64 — ensemble member x latitude [km]
"""
from __future__ import annotations
from pathlib import Path
import csv
import numpy as np

# Walk up from this script's directory until a sibling Europa2D/results/
# directory is found containing the mc_2d_*.npz archives.
_SCRIPT_DIR = Path(__file__).resolve().parent


def _find_archives_dir() -> Path:
    candidate = _SCRIPT_DIR
    for _ in range(6):
        probe = candidate / "Europa2D" / "results"
        if probe.is_dir() and any(probe.glob("mc_2d_*.npz")):
            return probe
        candidate = candidate.parent
    raise FileNotFoundError(
        "Could not locate Europa2D/results/ directory with mc_2d_*.npz archives. "
        "Pass --archives-dir explicitly or run from the project root."
    )


ARCHIVES_DIR = _find_archives_dir()
OUTPUT = _SCRIPT_DIR.parent / "results" / "dconv_threshold_fractions.csv"

SCENARIOS: dict[str, str] = {
    "uniform_transport": "mc_2d_uniform_transport_500.npz",
    "soderlund2014_equatorial_enhanced": "mc_2d_soderlund2014_equator_500.npz",
    "lemasquerier2023_polar_weak": "mc_2d_lemasquerier2023_polar_500.npz",
    "lemasquerier2023_polar_strong": "mc_2d_lemasquerier2023_polar_strong_500.npz",
}

THRESHOLDS_KM: list[float] = [8.0, 15.0]
EQUATOR_LAT_DEG = 0.0

LAT_KEY = "latitudes_deg"
DCONV_KEY = "D_conv_profiles"


def compute() -> list[list]:
    rows: list[list] = [
        ["scenario", "threshold_km", "fraction", "n_samples", "median_dconv_km"]
    ]
    for label, filename in SCENARIOS.items():
        path = ARCHIVES_DIR / filename
        if not path.exists():
            print(f"SKIP: {path} not found")
            continue
        d = np.load(path, allow_pickle=True)
        lats = d[LAT_KEY]             # shape (37,)
        dconv = d[DCONV_KEY]          # shape (500, 37)  members x lats

        # Axis-order check: axis-1 must match latitude count
        if dconv.shape[1] != lats.size:
            raise RuntimeError(
                f"Unexpected shape in {filename}: dconv={dconv.shape}, nlat={lats.size}. "
                "Expected (n_members, n_lats)."
            )

        eq_idx = int(np.argmin(np.abs(lats - EQUATOR_LAT_DEG)))
        eq_lat_actual = float(lats[eq_idx])
        eq_samples = dconv[:, eq_idx]                  # (500,)
        eq_samples = eq_samples[np.isfinite(eq_samples)]
        n = eq_samples.size
        med = float(np.median(eq_samples))

        for thr in THRESHOLDS_KM:
            frac = float((eq_samples >= thr).sum()) / n
            rows.append([label, thr, f"{frac:.3f}", n, f"{med:.2f}"])
            print(
                f"{label:>38}  D_conv>={thr:>4.1f} km  "
                f"P={frac:.3f}  N={n}  median={med:.2f} km  "
                f"(eq_lat={eq_lat_actual:.1f}°)"
            )
    return rows


def main() -> int:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    rows = compute()
    with OUTPUT.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    print(f"\nWrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
