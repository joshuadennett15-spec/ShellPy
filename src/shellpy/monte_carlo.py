"""Equilibrium shell thickness for one parameter set, and Monte Carlo ensembles of them."""
import multiprocessing as mp
import os
from collections import Counter
from functools import partial

import numpy as np
from scipy.stats import gaussian_kde

from shellpy import constants as c
from shellpy import physics, sampling
from shellpy.solver import ThermalSolver


def basal_heat_flux(p):
    """Ocean-to-ice heat flux (W/m^2): rocky-interior radiogenic plus silicate tidal heating."""
    return p['q_radiogenic'] + p['q_tidal_silicate']


def warm_start(p, q_basal, start):
    """Starting thickness (m) for start = 'thin' or 'thick'.

    Many parameter sets have two equilibria, a thinner conductive shell and a thicker
    convecting one, and which one a column reaches depends on where it starts.
    'thin' is 5 km, a shell growing from thin; 'thick' is the conductive estimate
    times 8, clipped to 5-100 km and to 95% of the hydrosphere, a shell thinning from thick.
    """
    if start == 'thin':
        return 5e3
    if start != 'thick':
        raise ValueError(f"start must be 'thin' or 'thick', got {start!r}")
    k = physics.conductivity((p['T_surf'] + c.MELT_TEMP) / 2)
    return float(np.clip(k * (c.MELT_TEMP - p['T_surf']) / q_basal * 8.0, 5e3, min(100e3, 0.95 * p['D_H2O'])))


def run_column(p, q_ocean, H0, *, nx=121, dt=1e12, **run_kw):
    """Run one column to equilibrium (see ThermalSolver.run); returns the solver."""
    solver = ThermalSolver(p, H0, nx=nx, dt=dt)
    solver.run(q_ocean, **run_kw)
    return solver


def column_outputs(solver):
    """Thickness and convection diagnostics of a finished column (km for lengths)."""
    s, H_km = solver.state, solver.H / 1e3
    out = {'H_km': H_km, 'D_cond_km': s.D_cond / 1e3, 'D_conv_km': s.D_conv / 1e3, 'Ra': s.Ra, 'Nu': s.Nu,
           'T_c': s.T_c, 'Ti': s.Ti, 'steps': solver.steps, 'converged': solver.converged}
    if s.D_conv > 0 and s.Ra < c.RA_CRIT:   # warm but subcritical sublayer: the whole shell conducts
        out.update(D_cond_km=H_km, D_conv_km=0.0, Nu=1.0)
    return out


def column_status(out, D_H2O):
    """'ok' for a shell at equilibrium over a liquid ocean, otherwise what happened instead.

    'frozen through': the shell reached the bottom of the hydrosphere, so no ocean is left.
    'melted through': the shell thinned to the 0.5 km floor the solver holds it at.
    """
    if out['H_km'] >= D_H2O / 1e3:
        return 'frozen through'
    if not out['converged']:
        return 'not converged'
    if out['H_km'] <= 0.5:
        return 'melted through'
    return 'ok'


def solve_column(p, q, start, **solver_kw):
    """Outputs and status of one column with ocean heat flux q (W/m^2), run from start."""
    if not q > 0:
        raise ValueError(f"basal heat flux must be positive, got {q}")
    solver_kw.setdefault('H_max', p['D_H2O'])
    out = column_outputs(run_column(p, q, warm_start(p, q, start), **solver_kw))
    out['status'] = column_status(out, p['D_H2O'])
    return out


def solve_1d(p, *, start, **solver_kw):
    """Equilibrium of one parameter set reached from start ('thin' or 'thick', see warm_start):
    outputs plus a 'status' (see column_status)."""
    return solve_column(p, basal_heat_flux(p), start, **solver_kw)


def sample_1d(i, seed):
    """Inputs of sample i of a 1D ensemble."""
    return sampling.sample_parameters(np.random.default_rng(seed + i))


def _run_one(i, seed, sample, solve, solve_kw):
    inputs = sample(i, seed)
    try:
        return inputs, solve(inputs, **solve_kw)
    except Exception as e:   # recorded per sample and counted by run_ensemble
        return inputs, {'status': f'error: {type(e).__name__}: {e}'}


def run_ensemble(n, seed, *, sample=sample_1d, solve=solve_1d, workers=None, **solve_kw):
    """Run n samples in parallel: inputs = sample(i, seed), outputs = solve(inputs, **solve_kw).

    Returns a dict of arrays with one row per sample: every input, every output
    and 'status'. Every row keeps its outputs whatever its status; only rows that
    raised an error have NaN outputs.
    """
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    job = partial(_run_one, seed=seed, sample=sample, solve=solve, solve_kw=solve_kw)
    with mp.Pool(workers) as pool:
        rows = pool.map(job, range(n), chunksize=max(1, n // (workers * 4)))
    inputs, outs = zip(*rows)
    table = {k: np.array([x[k] for x in inputs]) for k in inputs[0]}
    statuses = Counter(o['status'] for o in outs)
    for status, count in statuses.items():
        if status != 'ok':
            print(f"{count}/{n} samples: {status}")
    template = next((o for o in outs if not o['status'].startswith('error')), None)
    if template is None:
        raise RuntimeError(f"no sample succeeded: {dict(statuses)}")
    for k, v in template.items():
        if k != 'status':
            nan = np.full(np.shape(v), np.nan)
            table[k] = np.array([o.get(k, nan) for o in outs])
    table['status'] = np.array([o['status'] for o in outs])
    return table


def summarize(H):
    """Most likely value (KDE mode), median, mean and 1-sigma percentiles of thicknesses H."""
    H = np.asarray(H)
    grid = np.linspace(H.min(), H.max(), 300)
    return {'mode': float(grid[np.argmax(gaussian_kde(H)(grid))]), 'median': float(np.median(H)),
            'mean': float(np.mean(H)), 'p16': float(np.percentile(H, 15.87)), 'p84': float(np.percentile(H, 84.13))}
