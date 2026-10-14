"""Latitude-resolved shell: one independent column per latitude, equator (phi = 0) to pole (phi = pi/2).

Surface temperature, tidal strain and ocean heat flux vary with latitude; every other
shell property is shared by all columns. Heat flow between latitudes is neglected: in
ShellPy v1 it changed the equilibrium thickness by under 0.1 km.
"""
import numpy as np

from shellpy import constants as c
from shellpy import physics, sampling
from shellpy.monte_carlo import column_outputs
from shellpy.solver import ThermalSolver

# Ocean heat-flux scenarios: (pattern, q*), with q* the Lemasquerier et al. (2023) contrast parameter
SCENARIOS = {
    'uniform_transport': ('uniform', 0.0),                    # Ashkenazy & Tziperman (2021)
    'soderlund2014_equator': ('equator_enhanced', 0.45),      # Soderlund et al. (2014)
    'lemasquerier2023_polar': ('polar_enhanced', 0.455),      # Lemasquerier et al. (2023), mantle tidal fraction 0.5
    'lemasquerier2023_polar_strong': ('polar_enhanced', 0.819),   # same, mantle tidal fraction 0.9
}

SHARED = ('d_grain', 'd_del', 'D0v', 'D0b', 'mu_ice', 'D_H2O', 'Q_v', 'Q_b',
          'H_rad', 'f_porosity', 'f_salt', 'T_phi', 'B_k')


def surface_temperature(phi, T_eq, T_floor):
    """Annual-mean surface temperature (K); exponent 1.25 on cos(phi) fits Ashkenazy (2019)."""
    return ((T_eq**4 - T_floor**4) * np.cos(phi) ** 1.25 + T_floor**4) ** 0.25


def tidal_strain(phi, eps_eq, eps_pole):
    """Tidal strain amplitude, eps^2 proportional to 1 + c sin^2(phi) (Beuthe 2013 whole-shell pattern)."""
    return eps_eq * np.sqrt(1.0 + ((eps_pole / eps_eq) ** 2 - 1.0) * np.sin(phi) ** 2)


def ocean_heat_flux(phi, q_mean, scenario):
    """Ocean heat flux (W/m^2) at latitude phi; the area-weighted mean is q_mean."""
    pattern, q_star = SCENARIOS[scenario]
    if pattern == 'uniform':
        return np.full_like(np.asarray(phi, dtype=float), q_mean)
    if pattern == 'polar_enhanced':
        a = 3.0 * q_star / (3.0 - q_star)
        return q_mean * (1.0 + a * np.sin(phi) ** 2) / (1.0 + a / 3.0)
    a = 3.0 * q_star / (3.0 - 2.0 * q_star)
    return q_mean * (1.0 + a * np.cos(phi) ** 2) / (1.0 + 2.0 * a / 3.0)


def sample_latitude_parameters(seed_sequence, q_tidal_scale=1.2):
    """Shared shell parameters plus the latitude forcing anchors for one 2D sample.

    q_tidal_scale multiplies the silicate tidal part of the basal flux (1.2 in the thesis runs).
    """
    shared_seq, latitude_seq = seed_sequence.spawn(2)
    p = sampling.sample_parameters(np.random.default_rng(shared_seq))
    rng = np.random.default_rng(latitude_seq)
    inputs = {k: p[k] for k in SHARED}
    inputs['T_eq'] = p['T_surf']
    inputs['T_floor'] = min(float(np.clip(rng.normal(46.0, 4.0), 38.0, 55.0)), p['T_surf'] - 1.0)
    inputs['epsilon_eq'] = float(np.clip(10 ** rng.normal(np.log10(6e-6), 0.15), 3e-6, 1.2e-5))
    inputs['epsilon_pole'] = float(np.clip(10 ** rng.normal(np.log10(1.2e-5), 0.15), 6e-6, 2.5e-5))
    rng.uniform(0.1, 0.9)   # v1 mantle-tidal-fraction draw; unused because every scenario fixes q*
    M_rock = (4.0 / 3.0) * np.pi * ((c.RADIUS - p['D_H2O']) ** 3) * c.SILICATE_DENSITY
    inputs['q_ocean_mean'] = (p['H_rad'] * M_rock) / c.AREA + q_tidal_scale * (p['P_tidal'] / c.AREA)
    return inputs


def sample_2d(i, seed, q_tidal_scale=1.2):
    """Inputs of sample i of a 2D ensemble."""
    return sample_latitude_parameters(np.random.SeedSequence([seed, i]), q_tidal_scale)


def solve_2d(inputs, *, scenario, n_lat=37, nx=31, dt=1e12, max_steps=1500, tol=1e-12):
    """Equilibrium thickness at n_lat latitudes. Outputs are arrays over latitude, plus 'status'."""
    phi = np.linspace(0, np.pi / 2, n_lat)
    T_s = surface_temperature(phi, inputs['T_eq'], inputs['T_floor'])
    eps = tidal_strain(phi, inputs['epsilon_eq'], inputs['epsilon_pole'])
    q = ocean_heat_flux(phi, inputs['q_ocean_mean'], scenario)
    H0 = float(np.clip(physics.conductivity(190.0) * 170.0 / inputs['q_ocean_mean'] * 1.5, 5e3, 80e3))
    solvers = [ThermalSolver({**inputs, 'T_surf': T_s[j], 'epsilon_0': eps[j]}, H0, nx=nx, dt=dt)
               for j in range(n_lat)]
    for _ in range(max_steps):   # all columns step together until the slowest has stopped moving
        if max([abs(s.step(q[j])) for j, s in enumerate(solvers)]) < tol:
            break
    rows = [column_outputs(s) for s in solvers]
    out: dict = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    out['latitude_deg'] = np.degrees(phi)
    H = out['H_km']
    ok = (H > 0.5) & (H < inputs['D_H2O'] / 1e3 * 0.99) & (H < 200)
    out['n_interpolated'] = int(np.sum(~ok))
    if not ok.all():   # fill non-physical columns from their neighbours
        out['H_km'] = np.interp(out['latitude_deg'], out['latitude_deg'][ok], H[ok]) if ok.any() else H
    out['status'] = 'ok' if ok.sum() >= n_lat * 0.5 else 'rejected: thickness'
    return out
