"""Latitude-resolved shell: one independent column per latitude, equator (phi = 0) to pole (phi = pi/2).

Surface temperature, tidal strain and ocean heat flux vary with latitude; every other
shell property is shared by all columns. A 2D sample is the 1D sample with the same index,
spread over latitude: its area-weighted mean surface temperature is the 1D T_surf, its
area-weighted mean squared strain (tidal heating scales with strain^2) is the 1D epsilon_0^2,
and its area-weighted mean ocean heat flux is the 1D basal heat flux. Howell (2021) gives
T_surf and epsilon_0 as global means. Heat flow between latitudes is neglected: in
ShellPy v1 it changed the equilibrium thickness by under 0.1 km.
"""
import numpy as np
from scipy.integrate import trapezoid
from scipy.optimize import brentq

from shellpy.monte_carlo import basal_heat_flux, sample_1d, solve_column

# Ocean heat-flux scenarios: (pattern, q*), with q* the Lemasquerier et al. (2023) contrast parameter
SCENARIOS = {
    'uniform_transport': ('uniform', 0.0),                    # Ashkenazy & Tziperman (2021)
    'soderlund2014_equator': ('equator_enhanced', 0.40),      # Soderlund et al. (2014): flux varies ~40% with latitude
    'lemasquerier2023_polar': ('polar_enhanced', 0.455),      # Lemasquerier et al. (2023), mantle tidal fraction 0.5
    'lemasquerier2023_polar_strong': ('polar_enhanced', 0.819),   # same, mantle tidal fraction 0.9
}

STRAIN_RATIO = 2.0   # pole / equator tidal strain (v1, citing Tobie et al. 2003; not yet checked against the paper)

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


def equator_temperature(T_mean, T_floor):
    """Equatorial temperature whose surface_temperature profile has area-weighted mean T_mean."""
    phi = np.linspace(0, np.pi / 2, 2001)
    mean = lambda T_eq: trapezoid(surface_temperature(phi, T_eq, T_floor) * np.cos(phi), phi)
    return brentq(lambda T_eq: mean(T_eq) - T_mean, T_mean, 3 * T_mean)


def sample_2d(i, seed):
    """Inputs of sample i of a 2D ensemble: sample i of the 1D ensemble plus the polar surface
    temperature, and the latitude anchors that keep the 1D values as area-weighted means."""
    p = sample_1d(i, seed)
    rng = np.random.default_rng(np.random.SeedSequence([seed, i]))
    T_floor = min(float(np.clip(rng.normal(46.0, 4.0), 38.0, 55.0)), p['T_surf'] - 1.0)   # Ashkenazy (2019) pole
    eps_eq = p['epsilon_0'] / np.sqrt(1.0 + (STRAIN_RATIO**2 - 1.0) / 3.0)   # mean of sin^2 over area is 1/3
    return {**p, 'T_floor': T_floor, 'T_eq': equator_temperature(p['T_surf'], T_floor),
            'epsilon_eq': eps_eq, 'epsilon_pole': STRAIN_RATIO * eps_eq}


def solve_2d(inputs, *, scenario, start, n_lat=37, **solver_kw):
    """Equilibrium thickness at n_lat latitudes, each column reached from start ('thin' or
    'thick', see warm_start). Outputs are arrays over latitude, including 'column_status'
    (see column_status); 'status' is 'ok' when every column is, else the columns' problems.
    """
    phi = np.linspace(0, np.pi / 2, n_lat)
    T_s = surface_temperature(phi, inputs['T_eq'], inputs['T_floor'])
    eps = tidal_strain(phi, inputs['epsilon_eq'], inputs['epsilon_pole'])
    q = ocean_heat_flux(phi, basal_heat_flux(inputs), scenario)
    columns = [{**inputs, 'T_surf': T_s[j], 'epsilon_0': eps[j]} for j in range(n_lat)]
    rows = [solve_column(p, q[j], start, **solver_kw) for j, p in enumerate(columns)]
    out: dict = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    out['latitude_deg'] = np.degrees(phi)
    out['column_status'] = out.pop('status')
    out['status'] = ', '.join(sorted(set(out['column_status']) - {'ok'})) or 'ok'
    return out
