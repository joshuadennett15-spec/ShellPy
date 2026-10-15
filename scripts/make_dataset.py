"""Run the 1D or 2D Monte Carlo ensemble from both starts and save it in chunks, resumably.

    python scripts/make_dataset.py 1d 15000
    python scripts/make_dataset.py 2d 500 --scenario soderlund2014_equator

Each chunk is results/<name>/<start>_<first>-<last>.npz: one row per sample with every input,
every output and the labels (status, interior_melting, outside_dv_calibration, H_spread_km),
plus the git commit, seed and start. Chunks that exist are skipped, so a stopped run resumes.
"""
import argparse
import subprocess
from functools import partial
from pathlib import Path

import numpy as np

from shellpy.latitude import SCENARIOS, sample_2d, solve_2d
from shellpy.monte_carlo import run_ensemble, sample_1d, solve_1d


def shifted(i, seed, offset, sample):
    return sample(i + offset, seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('kind', choices=['1d', '2d'])
    ap.add_argument('n', type=int)
    ap.add_argument('--seed', type=int, default=10042)
    ap.add_argument('--scenario', choices=list(SCENARIOS))
    ap.add_argument('--starts', nargs='+', default=['thin', 'thick'], choices=['thin', 'thick'])
    ap.add_argument('--chunk', type=int, default=1000)
    ap.add_argument('--workers', type=int, help='parallel processes (default: all cores but one)')
    ap.add_argument('--out', type=Path, default=Path('results'))
    a = ap.parse_args()
    if (a.kind == '2d') != (a.scenario is not None):
        ap.error('--scenario is required for 2d and only for 2d')

    repo = Path(__file__).resolve().parents[1]
    commit = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=repo, capture_output=True, text=True, check=True).stdout.strip()
    if subprocess.run(['git', 'status', '--porcelain', 'src'], cwd=repo, capture_output=True, text=True, check=True).stdout:
        raise SystemExit('src/ has uncommitted changes: commit first so the data can be traced to a commit')
    sample, solve, kw = (sample_1d, solve_1d, {}) if a.kind == '1d' else (sample_2d, solve_2d, {'scenario': a.scenario})
    folder = a.out / (f'{a.kind}_{a.n}_seed{a.seed}' + (f'_{a.scenario}' if a.scenario else ''))
    folder.mkdir(parents=True, exist_ok=True)
    for start in a.starts:
        for first in range(0, a.n, a.chunk):
            m = min(a.chunk, a.n - first)
            path = folder / f'{start}_{first:06d}-{first + m - 1:06d}.npz'
            if path.exists():
                continue
            table = run_ensemble(m, a.seed, sample=partial(shifted, offset=first, sample=sample), solve=solve,
                                 workers=a.workers, start=start, **kw)
            table['sample_index'] = np.arange(first, first + m)
            np.savez(path, **table, commit=commit, seed=a.seed, start=start, scenario=str(a.scenario))
            print(f'{path}: {dict(zip(*np.unique(table["status"], return_counts=True)))}', flush=True)


if __name__ == '__main__':
    main()
