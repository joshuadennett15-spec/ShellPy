"""Latitude-resolved shell: one independent column per latitude, equator (phi = 0) to pole (phi = pi/2).

Surface temperature, tidal strain and ocean heat flux vary with latitude; every other
shell property is shared by all columns. Heat flow between latitudes is neglected: in
ShellPy v1 it changed the equilibrium thickness by under 0.1 km.
"""
import numpy as np

from shellpy import sampling
from shellpy.monte_carlo import solve_column

# Ocean heat-flux scenarios: (pattern, q*), with q* the Lemasquerier et al. (2023) contrast parameter
SCENARIOS = {
    'uniform_transport': ('uniform', 0.0),                    # Ashkenazy & Tziperman (2021)
    'soderlund2014_equator': ('equator_enhanced', 0.45),      # Soderlund et al. (2014)
    'lemasquerier2023_polar': ('polar_enhanced', 0.455),      # Lemasquerier et al. (2023), mantle tidal fraction 0.5
    'lemasquerier2023_polar_strong': ('polar_enhanced', 0.819),   # same, mantle tidal fraction 0.9
}

SHARED = ('d_grain', 'd_del', 'D0v', 'D0b', 'Q_v', 'Q_b', 'mu_ice', 'D_H2O', 'f_porosity',
          'q_radiogenic', 'q_tidal_silicate')


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


def sample_latitude_parameters(seed_sequence):
    """Shared shell parameters plus the latitude forcing anchors for one 2D sample."""
    shared_seq, latitude_seq = seed_sequence.spawn(2)
    p = sampling.sample_parameters(np.random.default_rng(shared_seq))
    rng = np.random.default_rng(latitude_seq)
    inputs = {k: p[k] for k in SHARED}
    inputs['T_eq'] = p['T_surf']
    inputs['T_floor'] = min(float(np.clip(rng.normal(46.0, 4.0), 38.0, 55.0)), p['T_surf'] - 1.0)
    inputs['epsilon_eq'] = float(np.clip(10 ** rng.normal(np.log10(6e-6), 0.15), 3e-6, 1.2e-5))
    inputs['epsilon_pole'] = float(np.clip(10 ** rng.normal(np.log10(1.2e-5), 0.15), 6e-6, 2.5e-5))
    return inputs


def sample_2d(i, seed):
    """Inputs of sample i of a 2D ensemble."""
    return sample_latitude_parameters(np.random.SeedSequence([seed, i]))


def solve_2d(inputs, *, scenario, start, n_lat=37, q_tidal_scale=1.2, **solver_kw):
    """Equilibrium thickness at n_lat latitudes, each column reached from start ('thin' or
    'thick', see warm_start). Outputs are arrays over latitude, including 'column_status'
    (see column_status); 'status' is 'ok' when every column is, else the columns' problems.

    q_tidal_scale multiplies the silicate tidal part of the basal flux (1.2 in the thesis runs).
    """
    phi = np.linspace(0, np.pi / 2, n_lat)
    T_s = surface_temperature(phi, inputs['T_eq'], inputs['T_floor'])
    eps = tidal_strain(phi, inputs['epsilon_eq'], inputs['epsilon_pole'])
    q_mean = inputs['q_radiogenic'] + q_tidal_scale * inputs['q_tidal_silicate']
    q = ocean_heat_flux(phi, q_mean, scenario)
    columns = [{**inputs, 'T_surf': T_s[j], 'epsilon_0': eps[j]} for j in range(n_lat)]
    rows = [solve_column(p, q[j], start, **solver_kw) for j, p in enumerate(columns)]
    out: dict = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    out['latitude_deg'] = np.degrees(phi)
    out['column_status'] = out.pop('status')
    out['status'] = ', '.join(sorted(set(out['column_status']) - {'ok'})) or 'ok'
    return out
