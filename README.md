# ShellPy

A thermal model of Europa's ice shell. Given the shell's material properties and the
heat reaching it from the ocean, ShellPy finds the equilibrium shell thickness, the
thickness of its conductive lid and convecting sublayer, and its temperature profile.
Monte Carlo ensembles over the uncertain properties turn this into a probability
distribution for the shell thickness, at one location (1D) or from equator to pole (2D).

> **Version 2 is under active development.** The dissertation version (Dennett 2026,
> University of Manchester) is preserved at tag [`v1.0.1`](https://github.com/joshuadennett15-spec/ShellPy/tree/v1.0.1).
> Version 2 fixes problems found in v1, listed in [CHANGELOG.md](CHANGELOG.md), so its
> numbers differ from the thesis.

## Install

```bash
pip install -e .            # numpy and scipy only
pip install -e ".[dev]"     # plus pytest
```

## Use

```python
import numpy as np
from shellpy.sampling import sample_parameters
from shellpy.monte_carlo import solve_1d, run_ensemble

p = sample_parameters(np.random.default_rng(0))   # one draw from the priors
print(solve_1d(p, start='thin'))                   # thickness, lid, Ra, Nu, ...

if __name__ == "__main__":                         # ensembles run in parallel processes
    for start in ('thin', 'thick'):                # one table per start, same inputs
        table = run_ensemble(1000, seed=10042, start=start)   # one row per sample: inputs, outputs, status
        np.savez(f"ensemble_{start}.npz", **table)
```

Every solve takes `start='thin'` (a shell growing from 5 km) or `start='thick'` (thinning from
a thick guess). With the DV2021 convection closure nearly every parameter set reaches the same
equilibrium from both; run both to see the few that do not (at the convection onset).
Each row carries labels: `status` ('ok', 'frozen through', 'melted through', 'not converged',
or an error), `interior_melting`, `outside_dv_calibration` and `H_spread_km`.
```

Latitude-resolved runs use `shellpy.latitude`:

```python
from shellpy.latitude import sample_2d, solve_2d
table = run_ensemble(500, seed=42, sample=sample_2d, solve=solve_2d, scenario="lemasquerier2023_polar", start="thin")
```

## Layout

| Module | Contents |
|---|---|
| `constants.py` | Physical constants (SI units) |
| `physics.py` | Conductivity, density, heat capacity, melting point, viscosity, tidal heating, Stefan condition |
| `convection.py` | Stagnant-lid convection: lid temperature, Rayleigh and Nusselt numbers, effective conductivity |
| `solver.py` | Time-marching solver for one shell column |
| `sampling.py` | Prior distributions of the shell parameters |
| `monte_carlo.py` | Equilibrium of one parameter set; parallel ensembles |
| `latitude.py` | Equator-to-pole runs: latitude-dependent forcing on independent columns |

Run the tests with `pytest`.

## Citation

See [CITATION.cff](CITATION.cff). Licensed under the MIT licence.
