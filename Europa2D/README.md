# Europa2D

Axisymmetric latitude-column extension of the one-dimensional thermal-shell
model. Each latitude is represented by an independent radial thermal column
sharing the rheology, convection closure, and Monte Carlo sampling
infrastructure of the one-dimensional code; the columns are coupled by a
finite-volume latitudinal diffusion operator on a metric-corrected grid in
$\phi$. Surface temperature, tidal strain rate, and ocean heat flux are
prescribed as latitude profiles drawn from the literature scenarios in
`src/literature_scenarios.py`.

## Layout

```text
Europa2D/
├── src/         Two-dimensional model
├── scripts/     Run and figure-generation entry points
├── tests/       Unit and regression tests
├── results/     500-member Monte Carlo archives (NPZ)
└── figures/     Final published figures (PDF and PNG)
```

The two-dimensional package depends on the radial solver and samplers in
the sibling `Europa1D/src/` directory. The path-injection at the top of
`src/__init__.py` makes those modules importable under their original
names (`from Solver import Thermal_Solver`, `from constants import
Planetary, Thermal`, and so on).

## `src/`

| Module | Role |
|--------|------|
| `axial_solver.py`         | Time-stepped 2D thermal solver coupling radial columns through latitudinal diffusion. |
| `convection_2d.py`        | Stagnant-lid convection closure applied per column. |
| `latitude_profile.py`     | Surface-temperature, tidal-strain, and ocean-heat-flux latitude profiles. |
| `latitude_sampler.py`     | Monte Carlo prior sampler extending the audited 1D sampler with latitude-distributed fluxes. |
| `literature_scenarios.py` | Named ocean-heat-transport scenarios (uniform, Soderlund 2014 equator-enhanced, Lemasquerier 2023 polar). |
| `monte_carlo_2d.py`       | Parallel Monte Carlo runner producing the per-scenario ensembles. |
| `attribution_cases.py`    | Deterministic test cases used when checking convergence and physics flags. |
| `profile_diagnostics.py`  | Latitude-band statistics and ocean-pattern metadata helpers. |

## `scripts/`

| Script | Output |
|--------|--------|
| `run_2d_mc.py`                    | Four-scenario 500-member Monte Carlo ensembles, written to `results/mc_2d_<scenario>_500.npz`. |
| `plot_mc_500_suite.py`            | `fig_2d_shell_structure`, `fig_2d_violin_dcond`, `fig_2d_lid_fraction`, `fig_2d_convection_diagnostics`, `fig_2d_temperature_profiles`. |
| `bayes_factor_latitude_sweep.py`  | Latitude sweep of the Juno-conditioned Bayes factor, written to `results/bayes_factor_latitude_sweep.npz`. |
| `plot_bayes_factor_latitude.py`   | `fig_bayes_factor_latitude`. |
| `plot_scenario_separability.py`   | `fig_scenario_separation` and `fig_precision_threshold`. |
| `pub_style.py`                    | Shared figure style and palette. |

The figure scripts read NPZ archives produced by the runners, so the
runners must be executed first. Output figures are written to
`figures/thesis/` (created on demand) as PDF and PNG.

## Running the model

```bash
cd Europa2D
python scripts/run_2d_mc.py            # generates the four scenario ensembles
python scripts/plot_mc_500_suite.py    # plots the five MC suite figures
python scripts/bayes_factor_latitude_sweep.py
python scripts/plot_bayes_factor_latitude.py
python scripts/plot_scenario_separability.py
```

All seeds are exposed as command-line arguments or as named constants at the
top of each runner; the figures in the thesis use the default values.
