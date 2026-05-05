# ShellPy

This repository accompanies the dissertation *xxxx* (Joshua Dennett,
University of Manchester, 2026) and serves as its code- and
data-availability statement. It bundles the one-dimensional radial
thermal-shell model and the two-dimensional axisymmetric extension on
which the methodology and results chapters rely, together with the Monte
Carlo archives and final figures reproduced in the thesis.

The 1D package (`Europa1D/`) provides the radial finite-volume solver, the
audited Monte Carlo sampling framework, the Sobol sensitivity workflow, and
the Bayesian Juno-conditioning pipeline that underpin Chapter 3. The 2D
package (`Europa2D/`) couples a set of independent radial thermal columns
through a finite-volume latitudinal diffusion operator and applies the
1D rheology, convection closure, and sampler infrastructure column by
column; it produces the headline figures of Chapter 4.

## Repository layout

```text
ShellPy/
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── .gitignore
├── Europa1D/
│   ├── README.md
│   ├── src/        Radial solver, physics, samplers, Monte Carlo, Sobol workflow
│   ├── scripts/    1D run and figure-generation entry points
│   ├── tests/      Unit and regression tests
│   ├── results/    NPZ archives and summary tables underlying Chapter 3
│   └── figures/    Published 1D figures (PDF and PNG)
└── Europa2D/
    ├── README.md
    ├── src/        Two-dimensional model and parameter samplers
    ├── scripts/    2D run and figure-generation entry points
    ├── tests/      Unit and regression tests
    ├── results/    500-member 2D Monte Carlo archives used in Chapter 4
    └── figures/thesis/  Final published 2D figures (PDF and PNG)
```

The `results/` and `figures/` directories of both packages are tracked in
this repository so that the figures and their underlying Monte Carlo
archives are available without re-running the model. The scripts described
below and in each package's README regenerate the outputs from scratch.

## Requirements

The model is written in standard scientific Python and depends only on
`numpy`, `scipy`, and `matplotlib`. A working installation can be obtained
with

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The code has been run against Python 3.11 and 3.12 on Windows, Linux, and
macOS. No compiled extensions are required. The 2D package imports the
1D modules via a relative path from `Europa2D/src/` to `Europa1D/src/`, so
the two directories must be checked out together.

## Reproducing the thesis figures

The 1D and 2D figure suites are independent. The full thesis sequence is

```bash
# Chapter 3 — one-dimensional results
cd Europa1D
python scripts/run_andrade_15k.py            # 15k-sample headline ensemble
python scripts/run_sobol_suite.py            # Sobol design and evaluation
python scripts/sobol_analysis.py             # first/total-order indices
python scripts/plot_sobol.py                 # Sobol tornado figure
python scripts/bayesian_inversion_juno.py    # Juno reweighting (global)
python scripts/bayesian_refit_equatorial.py  # Juno refit (equatorial)
python scripts/run_endmember_suite.py        # endmember scenarios
python scripts/run_equatorial_suite.py
python scripts/run_polar_suite.py
cd ..

# Chapter 4 — two-dimensional results
cd Europa2D
python scripts/run_2d_mc.py                       # four-scenario ensembles
python scripts/plot_mc_500_suite.py               # five MC suite figures
python scripts/bayes_factor_latitude_sweep.py     # Bayes factor sweep
python scripts/plot_bayes_factor_latitude.py
python scripts/plot_scenario_separability.py     # scenario-separation panels
```

A single 2D Monte Carlo ensemble takes of order an hour on a recent
eight-core desktop; the full 2D figure suite takes roughly three hours
from a clean checkout. The 1D ensembles are roughly an order of magnitude
cheaper. Random seeds are passed explicitly to every Monte Carlo runner,
so re-running the same script on the same inputs reproduces the published
figures bit-for-bit.

## Scope

The repository contains the model source, figure-generation scripts, unit
and regression tests, the Monte Carlo archives used to draw the thesis
figures, and the final published figures in PDF and PNG form. The thesis
manuscript itself and intermediate research notes are kept in a separate
working tree and are not part of this release.
