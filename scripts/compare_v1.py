"""Compare a v2 1D ensemble (scripts/make_dataset.py) with the v1.0.1 thesis ensemble.

    python scripts/compare_v1.py results/1d_15000_seed10042

Writes <folder>/compare_v1.png (thickness and lid-thickness distributions) and
<folder>/compare_v1.md (summary table). v2 distributions use the rows whose status is 'ok'.
"""
import subprocess
import sys
from io import BytesIO
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from shellpy.monte_carlo import summarize

V1_ARCHIVE = 'v1.0.1:Europa1D/results/mc_15000_optionA_v2_andrade.npz'
COLORS = {'v1.0.1': '#2a78d6', 'v2 thick start': '#eb6834', 'v2 thin start': '#1baf7a'}


def load_v2(folder, start):
    chunks = [np.load(f) for f in sorted(Path(folder).glob(f'{start}_*.npz'))]
    if not chunks:
        raise SystemExit(f'no {start} chunks in {folder}')
    keys = [k for k in chunks[0].files if chunks[0][k].ndim == 1]
    return {k: np.concatenate([c[k] for c in chunks]) for k in keys}, str(chunks[0]['commit'])


def load_v1(repo):
    blob = subprocess.run(['git', 'show', V1_ARCHIVE], cwd=repo, capture_output=True, check=True).stdout
    d = np.load(BytesIO(blob))
    return {'H_km': d['thicknesses_km'], 'D_cond_km': d['D_cond_km'], 'D_conv_km': d['D_conv_km']}


def row(name, d, statuses=None):
    s = summarize(d['H_km'])
    counts = ', '.join(f'{k} {v}' for k, v in statuses.items()) if statuses else 'not recorded'
    return (f"| {name} | {len(d['H_km'])} | {counts} | {s['mode']:.1f} | {s['median']:.1f} | {s['p16']:.1f}-{s['p84']:.1f} "
            f"| {np.mean(d['D_conv_km'] > 0):.0%} | {np.median(d['D_cond_km']):.1f} |")


def main(folder):
    repo = Path(__file__).resolve().parents[1]
    sets, lines = {'v1.0.1': (load_v1(repo), None)}, []
    for start in ('thick', 'thin'):
        d, commit = load_v2(folder, start)
        ok = d['status'] == 'ok'
        statuses = dict(zip(*np.unique(d['status'], return_counts=True)))
        sets[f'v2 {start} start'] = ({k: v[ok] for k, v in d.items()}, statuses)
    lines += [f'v2 commit {commit}; v1 archive {V1_ARCHIVE}', '',
              '| Ensemble | n used | Status | Mode (km) | Median (km) | 16-84% (km) | Convecting | Median lid (km) |',
              '|---|---|---|---|---|---|---|---|']
    lines += [row(name, d, st) for name, (d, st) in sets.items()]
    (Path(folder) / 'compare_v1.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, key, label in zip(axes, ('H_km', 'D_cond_km'), ('Ice shell thickness (km)', 'Conductive lid thickness (km)')):
        grid = np.linspace(0, np.percentile(np.concatenate([d[key] for d, _ in sets.values()]), 99.5), 400)
        for name, (d, _) in sets.items():
            style = '--' if 'thin' in name else '-'
            y = gaussian_kde(d[key])(grid)
            ax.plot(grid, y, style, color=COLORS[name], lw=2, label=name)
            if 'thin' not in name:   # thin and thick curves nearly coincide; the legend names the dashed one
                i = int(np.argmax(y))
                ax.annotate(name, (grid[i], y[i]), textcoords='offset points', xytext=(6, 4), fontsize=8, color='#333333')
        ax.set_xlabel(label)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.25, lw=0.5)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel('Probability density (1/km)')
    axes[0].legend(frameon=False, fontsize=9)
    fig.savefig(Path(folder) / 'compare_v1.png', dpi=200)
    print('\n'.join(lines))


if __name__ == '__main__':
    main(sys.argv[1])
