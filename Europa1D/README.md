# Europa1D

One-dimensional radial thermal-shell model for Europa, with an
audited Monte Carlo sampling framework, a Sobol sensitivity workflow, and
a Bayesian Juno-conditioning pipeline. The package supplies the radial
solver and sampler infrastructure on which the two-dimensional axisymmetric
extension in `Europa2D/` builds.

## Layout

```text
Europa1D/
├── src/        Radial solver, physics, samplers, Monte Carlo, Sobol workflow
├── scripts/    Run and figure-generation entry points
├── tests/      Unit and regression tests
├── results/    NPZ archives and summary tables underlying the thesis figures
└── figures/    Published figures (PDF and PNG)
```

## `src/`

| Module | Role |
|--------|------|
| `constants.py`              | Planetary, thermal, and rheological dataclasses; type aliases. |
| `ConfigManager.py`          | Loader for the JSON parameter file used by `constants`. |
| `config.json`               | Default parameter values backing `ConfigManager`. |
| `Physics.py`                | `IcePhysics`: viscosity, conductivity, capacity, tidal dissipation. |
| `Convection.py`             | Parameterised stagnant-lid convection (`IceConvection`). |
| `Boundary_Conditions.py`    | Surface boundary-condition helpers. |
| `Solver.py`                 | Transient one-dimensional finite-volume thermal solver. |
| `batched_solver.py`         | Vectorised solver used by Sobol and large Monte Carlo runs. |
| `wattmeter.py`              | Equilibrium heat-flux closure used by the basal-flux samplers. |
| `Monte_Carlo.py`            | Monte Carlo runner and the Howell parameter sampler. |
| `audited_sampler.py`        | Literature-audited prior sampler (`AuditedShellSampler`). |
| `audited_endmember_sampler.py` | Endmember scenario sampler. |
| `audited_equatorial_sampler.py` | Equatorial-column variant. |
| `audited_polar_sampler.py`  | Polar-column variant. |
| `audited_wide_priors_sampler.py` | Wide-prior smoke-test sampler. |
| `juno_constrained_sampler.py` | Juno-conditioned sampler used by the Bayesian inversion. |
| `regional_samplers.py`      | Regional parameter samplers (equator/pole). |
| `regional_samplers_500.py`  | 500-sample regional variant used in the published runs. |
| `budget_samplers.py`        | Heat-budget sampler variants. |
| `sobol_workflow.py`         | Saltelli/Sobol design and first/total-order index estimation. |
| `runtime_support.py`        | Numeric runtime configuration shared across the pipeline. |

## `scripts/`

The scripts directory groups three families of entry points:

- **Monte Carlo runners** (`run_*.py`) — generate the prior ensembles and
  scenario archives saved under `results/`. Examples include
  `run_andrade_15k.py`, `run_500_new_priors.py`, `run_endmember_suite.py`,
  `run_polar_suite.py`, and `run_equatorial_suite.py`.
- **Bayesian inversion** — `bayesian_inversion_juno.py` and
  `bayesian_refit_equatorial.py` reweight the prior draws against the Juno
  MWR shell-thickness observation, and `compute_dconv_thresholds.py` and
  `compute_dconv_thresholds_1d.py` evaluate the chaos-terrain consistency
  thresholds.
- **Figure generation** (`plot_*.py`, `generate_pub_figures*.py`) — read
  archives from `results/` and write the published figures into `figures/`.
  `pub_style.py` defines the shared figure style.

## `tests/`

Unit tests cover the Sobol physics plumbing, the wattmeter closure, the
DV2021 conductivity path, the GBS creep formula, the regional and
endmember samplers, the regression baselines, and the Tidal validation.
Run with `pytest -q` from the repository root.

## Reproducing the 1D thesis figures

```bash
cd Europa1D

# Headline 1D Monte Carlo ensemble (15k samples, Andrade rheology)
python scripts/run_andrade_15k.py

# Sobol sensitivity study (writes results/sobol/)
python scripts/run_sobol_suite.py
python scripts/sobol_analysis.py
python scripts/plot_sobol.py

# Bayesian Juno reweighting (writes results/ posteriors)
python scripts/bayesian_inversion_juno.py
python scripts/bayesian_refit_equatorial.py

# Endmember and regional scenario suites
python scripts/run_endmember_suite.py
python scripts/run_equatorial_suite.py
python scripts/run_polar_suite.py
```

Random seeds are passed explicitly to every Monte Carlo runner; re-running
the same script on the same inputs reproduces the published figures
bit-for-bit.
