"""Compute equatorial D_conv threshold fractions for the 1D scenario archives.

Sibling of compute_dconv_thresholds.py. The 2D script processes 500-member
latitude-aware ensembles at the equator; this script processes the higher-N
1D scenario archives (10k or 15k samples, equatorial column only) so the
chaos-consistency claim can be reported with tighter Monte Carlo error.

Thresholds from Nimmo & Manga (2002):
  8 km  - coherent warm-ice diapirs
  15 km - widespread chaos coverage (~40% of surface)

Archives in Europa1D/results/eq_*_andrade.npz with key D_conv_km
of shape (n_iter,). Sample counts: baseline/depleted/moderate/strong = 10k,
depleted_strong = 15k.
"""
from __future__ import annotations
from pathlib import Path
import csv
import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
ARCHIVES_DIR = _SCRIPT_DIR.parent / "results"
OUTPUT = ARCHIVES_DIR / "dconv_threshold_fractions_1d.csv"

SCENARIOS: dict[str, str] = {
    "depleted_strong": "eq_depleted_strong_andrade.npz",
    "depleted":        "eq_depleted_andrade.npz",
    "baseline":        "eq_baseline_andrade.npz",
    "moderate":        "eq_moderate_andrade.npz",
    "strong":          "eq_strong_andrade.npz",
}

THRESHOLDS_KM: list[float] = [8.0, 15.0]
DCONV_KEY = "D_conv_km"


def compute() -> list[list]:
    rows: list[list] = [
        ["scenario", "threshold_km", "fraction", "se_pp",
         "n_samples", "median_dconv_km"]
    ]
    for label, filename in SCENARIOS.items():
        path = ARCHIVES_DIR / filename
        if not path.exists():
            print(f"SKIP: {path} not found")
            continue
        d = np.load(path, allow_pickle=True)
        dconv = np.asarray(d[DCONV_KEY])
        dconv = dconv[np.isfinite(dconv)]
        n = dconv.size
        med = float(np.median(dconv))

        for thr in THRESHOLDS_KM:
            k = int((dconv >= thr).sum())
            p = k / n
            # Wilson would be tighter near 0/1, but normal SE is fine here
            # (p stays well clear of bounds for these scenarios) and matches
            # the 2D CSV's reporting style.
            se_pp = 100.0 * float(np.sqrt(p * (1.0 - p) / n))
            rows.append([label, thr, f"{p:.4f}", f"{se_pp:.2f}",
                         n, f"{med:.2f}"])
            print(
                f"{label:>16}  D_conv>={thr:>4.1f} km  "
                f"P={p:.4f} ± {se_pp:.2f}pp  N={n}  median={med:.2f} km"
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
