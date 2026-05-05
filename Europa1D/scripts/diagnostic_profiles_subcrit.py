"""
Diagnostic Monte Carlo: 500 runs capturing full temperature + tidal heating profiles.

Saves equilibrium T(z), q_tidal(z), k_eff(z), and convection state for each valid sample
so we can visualise how internal heating varies with shell thickness.
"""
import sys, os
from math import gamma
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from runtime_support import configure_numeric_runtime, resolve_worker_count

configure_numeric_runtime()

import numpy as np
import multiprocessing as mp
from typing import Optional, Dict, Any

from pub_style import apply_style, label_panel, save_fig, add_minor_gridlines, figsize_double
from constants import Thermal, Planetary, Rheology, HeatFlux
from constants import Convection as ConvectionConstants
from Physics import IcePhysics
from Convection import IceConvection
from Solver import Thermal_Solver
from Boundary_Conditions import FixedTemperature
from Monte_Carlo import HowellParameterSampler, SolverConfig
from regional_samplers import EquatorParameterSampler, PoleParameterSampler

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
DEFAULT_DIAGNOSTIC_WORKERS = 15

apply_style()

class UpdatedPtidalSampler(HowellParameterSampler):
    """Howell sampler with P_tidal ~ U(150, 350) GW."""
    def sample(self):
        params = super().sample()
        params['P_tidal'] = self.rng.uniform(150e9, 350e9)
        return params

class UpdatedEquatorSampler(EquatorParameterSampler):
    """Equator sampler with P_tidal ~ U(150, 350) GW."""
    def sample(self):
        params = super().sample()
        params['P_tidal'] = self.rng.uniform(150e9, 350e9)
        return params

class UpdatedPoleSampler(PoleParameterSampler):
    """Pole sampler with P_tidal ~ U(150, 350) GW."""
    def sample(self):
        params = super().sample()
        params['P_tidal'] = self.rng.uniform(150e9, 350e9)
        return params


# Lookup table for sampler classes (picklable by name)
_SAMPLER_MAP = {
    "HowellParameterSampler":   HowellParameterSampler,
    "EquatorParameterSampler":  EquatorParameterSampler,
    "PoleParameterSampler":     PoleParameterSampler,
    "UpdatedPtidalSampler":     UpdatedPtidalSampler,
    "UpdatedEquatorSampler":    UpdatedEquatorSampler,
    "UpdatedPoleSampler":       UpdatedPoleSampler,
}


# ── worker ──────────────────────────────────────────────────────────────────
def _run_diagnostic_sample(args):
    """Run one MC sample and return full profiles."""
    sample_id, base_seed, sampler_name = args
    config = SolverConfig()          # default config
    sampler_cls = _SAMPLER_MAP[sampler_name]
    try:
        sampler = sampler_cls(seed=base_seed + sample_id)
        params = sampler.sample()

        T_surf = params['T_surf']
        D_H2O  = params['D_H2O']
        H_rad  = params['H_rad']
        P_tidal = params['P_tidal']

        # Basal flux (same as Monte_Carlo._run_single_sample)
        R_europa = Planetary.RADIUS
        R_rock   = R_europa - D_H2O
        A_surface = Planetary.AREA
        rho_rock  = 3500.0
        M_rock = (4.0 / 3.0) * np.pi * (R_rock ** 3) * rho_rock
        q_radiogenic = (H_rad * M_rock) / A_surface
        q_silicate_tidal = P_tidal / A_surface
        q_basal = q_radiogenic + q_silicate_tidal

        # Warm-start thickness
        T_melt = Thermal.MELT_TEMP
        k_mean = Thermal.conductivity((T_surf + T_melt) / 2)
        H_guess = np.clip((k_mean * (T_melt - T_surf)) / q_basal, 5e3, 100e3)

        surface_bc = FixedTemperature(temperature=T_surf)
        solver = Thermal_Solver(
            nx=config.nx,
            thickness=H_guess,
            dt=config.dt,
            total_time=config.total_time,
            coordinate_system=config.coordinate_system,
            surface_bc=surface_bc,
            rannacher_steps=config.rannacher_steps,
            use_convection=config.use_convection,
            physics_params=params,
        )

        # Run to equilibrium
        for step in range(config.max_steps):
            velocity = solver.solve_step(q_basal)
            if abs(velocity) < config.eq_threshold:
                break

        H_km = solver.H / 1000.0
        D_H2O_km = D_H2O / 1000.0
        if H_km <= 0.5 or H_km >= D_H2O_km * 0.99 or H_km > 200:
            return None

        # ── capture profiles ────────────────────────────────────────────
        profile_diag = solver.get_profile_diagnostics()
        T_profile = profile_diag['temperature_K']
        z_grid = profile_diag['depth_m']
        q_tidal = profile_diag['tidal_heating_W_m3']
        k_profile = profile_diag['effective_conductivity_W_mK']
        conv_diag = profile_diag['convection']

        # Viscosity profile (composite diffusion creep, same as used by solver)
        eta_profile = IcePhysics.composite_viscosity(
            T_profile,
            d_grain=params.get('d_grain'),
            d_del=params.get('d_del'),
            D0v=params.get('D0v'),
            D0b=params.get('D0b'),
            Q_diff=params.get('Q_v'),
            Q_gbs=params.get('Q_b'),
        )

        # Convection diagnostics
        conv = {}
        if conv_diag:
            conv = {
                'D_cond_km': conv_diag['D_cond_km'],
                'D_conv_km': conv_diag['D_conv_km'],
                'lid_fraction': conv_diag['lid_fraction'],
                'Ra': conv_diag['Ra'],
                'Nu': conv_diag['Nu'],
                'Nu_raw': conv_diag['Nu_raw'],
                'convection_ramp': conv_diag['convection_ramp'],
                'T_c': conv_diag['T_c'],
                'idx_c': conv_diag['idx_c'],
                'is_convecting': conv_diag['is_convecting'],
            }
        else:
            conv = {'D_cond_km': H_km, 'D_conv_km': 0.0,
                    'lid_fraction': 1.0, 'Ra': 0.0, 'Nu': 1.0,
                    'Nu_raw': 1.0, 'convection_ramp': solver.convection_ramp,
                    'T_c': 0.0, 'idx_c': solver.nx - 1,
                    'is_convecting': False}

        # Depth-integrated tidal heating (W/m²)
        q_tidal_integrated = np.trapezoid(q_tidal, z_grid)

        return {
            'H_km':      H_km,
            'T_profile': T_profile,       # (nx,) K
            'z_grid_km': z_grid / 1000.0, # (nx,) km
            'q_tidal':   q_tidal,         # (nx,) W/m³
            'k_eff':     k_profile,       # (nx,) W/(m·K)
            'eta':       eta_profile,     # (nx,) Pa·s
            'q_basal':   q_basal,         # W/m²
            'q_tidal_integrated': q_tidal_integrated,  # W/m²
            'params':    params,
            'conv':      conv,
        }
    except Exception as e:
        if sample_id < 3:
            import traceback
            traceback.print_exc()
        return None


# ── main ────────────────────────────────────────────────────────────────────
def run_diagnostics(n_samples=500, seed=44, n_workers=None, sampler_class=None):
    if n_workers is None:
        n_workers = DEFAULT_DIAGNOSTIC_WORKERS
    n_workers = resolve_worker_count(n_workers)
    if sampler_class is None:
        sampler_class = HowellParameterSampler
    sampler_name = sampler_class.__name__
    print(f"Running {n_samples} diagnostic samples "
          f"(sampler={sampler_name}, seed={seed}, workers={n_workers})")
    args = [(i, seed, sampler_name) for i in range(n_samples)]

    with mp.Pool(processes=n_workers) as pool:
        results = list(pool.imap_unordered(_run_diagnostic_sample, args, chunksize=10))

    valid = [r for r in results if r is not None]
    print(f"  Valid: {len(valid)} / {n_samples}")
    return valid


def save_diagnostics(valid, outpath):
    """Save diagnostic results to npz."""
    n = len(valid)
    nx = len(valid[0]['T_profile'])

    # Fixed-size arrays
    H_km       = np.array([r['H_km'] for r in valid])
    q_basal    = np.array([r['q_basal'] for r in valid])
    q_int      = np.array([r['q_tidal_integrated'] for r in valid])
    D_cond_km  = np.array([r['conv']['D_cond_km'] for r in valid])
    D_conv_km  = np.array([r['conv']['D_conv_km'] for r in valid])
    lid_frac   = np.array([r['conv']['lid_fraction'] for r in valid])
    Ra         = np.array([r['conv']['Ra'] for r in valid])
    Nu         = np.array([r['conv']['Nu'] for r in valid])
    Nu_raw     = np.array([r['conv']['Nu_raw'] for r in valid])
    convection_ramp = np.array([r['conv']['convection_ramp'] for r in valid])
    T_c        = np.array([r['conv']['T_c'] for r in valid])

    # Profiles: (n, nx) arrays
    T_profiles = np.array([r['T_profile'] for r in valid])
    z_profiles = np.array([r['z_grid_km'] for r in valid])
    q_profiles = np.array([r['q_tidal'] for r in valid])
    k_profiles = np.array([r['k_eff'] for r in valid])
    eta_profiles = np.array([r['eta'] for r in valid])

    # Key sampled params
    epsilon_0 = np.array([r['params']['epsilon_0'] for r in valid])
    d_grain   = np.array([r['params']['d_grain'] for r in valid])
    mu_ice    = np.array([r['params']['mu_ice'] for r in valid])
    Q_v       = np.array([r['params']['Q_v'] for r in valid])

    np.savez(outpath,
             H_km=H_km, q_basal=q_basal, q_tidal_integrated=q_int,
             D_cond_km=D_cond_km, D_conv_km=D_conv_km, lid_frac=lid_frac,
             Ra=Ra, Nu=Nu, Nu_raw=Nu_raw, convection_ramp=convection_ramp, T_c=T_c,
             T_profiles=T_profiles, z_profiles=z_profiles,
             q_profiles=q_profiles, k_profiles=k_profiles,
             eta_profiles=eta_profiles,
             epsilon_0=epsilon_0, d_grain=d_grain, mu_ice=mu_ice, Q_v=Q_v)
    print(f"  Saved to {outpath}")


# ── helper: percentile envelope on a common normalised grid ─────────────────
def _envelope(profiles, H_arr, mask, n_interp=100):
    """
    Interpolate profiles in *mask* onto a common normalised-depth grid
    and return median / 10-90 percentile bands.
    """
    zn = np.linspace(0, 1, n_interp)
    stack = []
    nx = profiles.shape[1]
    z_norm_nodes = np.linspace(0, 1, nx)
    for i in np.where(mask)[0]:
        stack.append(np.interp(zn, z_norm_nodes, profiles[i]))
    if len(stack) == 0:
        return zn, None, None, None
    stack = np.array(stack)
    return zn, np.median(stack, axis=0), np.percentile(stack, 10, axis=0), \
           np.percentile(stack, 90, axis=0)


def _andrade_reference_viscosity(mu_ref, omega):
    """Return the viscosity at peak dissipation for the current Andrade law."""
    alpha = Rheology.ANDRADE_ALPHA
    zeta = Rheology.ANDRADE_ZETA
    x = np.logspace(-6, 6, 4000)
    j_elastic = 1.0 / mu_ref
    andrade_term = np.clip(zeta * x, 1e-100, None)
    const_term = j_elastic * (andrade_term ** (-alpha)) * gamma(1.0 + alpha)
    j_real = j_elastic + const_term * np.cos(alpha * np.pi / 2.0)
    j_imag = j_elastic / x + const_term * np.sin(alpha * np.pi / 2.0)
    g_imag = j_imag / (j_real**2 + j_imag**2)
    x_opt = float(x[np.argmax(g_imag)])
    return x_opt * mu_ref / omega


# ── thickness bins used across all panels ───────────────────────────────────
BINS = [
    ("< 15 km",   0,  15, "#1b9e77"),
    ("15–30 km", 15,  30, "#d95f02"),
    ("30–50 km", 30,  50, "#7570b3"),
    ("50–80 km", 50,  80, "#e7298a"),
    ("> 80 km",  80, 999, "#66a61e"),
]


# ── plotting ────────────────────────────────────────────────────────────────
def plot_diagnostics(datapath, tag="global"):
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize, LogNorm
    from matplotlib.cm import ScalarMappable

    d = np.load(datapath)
    # Mask to keep only subcritical (conductive) models
    mask_subcrit = d['Nu'] == 1.0
    
    H     = d['H_km'][mask_subcrit]
    q_bas = d['q_basal'][mask_subcrit] * 1e3              # → mW/m²
    q_int = d['q_tidal_integrated'][mask_subcrit] * 1e3   # → mW/m²
    T_pr  = d['T_profiles'][mask_subcrit].copy()          # (n, nx)
    z_pr  = d['z_profiles'][mask_subcrit]                 # (n, nx) km
    q_pr  = d['q_profiles'][mask_subcrit] * 1e6           # → μW/m³
    k_pr  = d['k_profiles'][mask_subcrit]                 # W/(m·K)
    D_cond = d['D_cond_km'][mask_subcrit]
    D_conv = d['D_conv_km'][mask_subcrit]
    Nu     = d['Nu'][mask_subcrit]
    n = len(H)
    if n == 0:
        print(f"No subcritical models found in {datapath}!")
        return
    nx = T_pr.shape[1]

    # Enforce isothermal convective cores for visualization
    if 'T_c' in d:
        T_c = d['T_c']
        for i in range(n):
            if Nu[i] > 1.0 and D_cond[i] < H[i]:
                is_conv = z_pr[i] > D_cond[i]
                if np.any(is_conv):
                    T_base = T_pr[i, -1]
                    T_pr[i, is_conv] = T_c[i]
                    T_pr[i, -1] = T_base

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # ------------------------------------------------------------------
    # (a) Temperature profiles — binned median + 10-90 % band
    # ------------------------------------------------------------------
    ax = axes[0]
    for label, lo, hi, col in BINS:
        mask = (H >= lo) & (H < hi)
        zn, med, p10, p90 = _envelope(T_pr, H, mask)
        if med is None:
            continue
        ax.plot(med, zn, color=col, lw=2, label=f"{label} (N={mask.sum()})")
        ax.fill_betweenx(zn, p10, p90, color=col, alpha=0.15)
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("Normalised depth  z / H")
    ax.invert_yaxis()
    ax.set_title("(a) Temperature profiles — median ± 10-90 %")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(True, alpha=0.2)

    # ------------------------------------------------------------------
    # (b) Tidal heating q̇(T) — binned, LOG y-axis
    # ------------------------------------------------------------------
    ax = axes[1]
    # Interpolate each sample onto a common T grid
    T_common = np.linspace(100, 273, 200)
    for label, lo, hi, col in BINS:
        mask = (H >= lo) & (H < hi)
        stack = []
        for i in np.where(mask)[0]:
            # sort by T for safe interpolation
            order = np.argsort(T_pr[i])
            T_sorted = T_pr[i][order]
            q_sorted = q_pr[i][order]
            stack.append(np.interp(T_common, T_sorted, q_sorted, left=0, right=0))
        if len(stack) == 0:
            continue
        stack = np.array(stack)
        med = np.median(stack, axis=0)
        p10 = np.percentile(stack, 10, axis=0)
        p90 = np.percentile(stack, 90, axis=0)
        ax.plot(T_common, med, color=col, lw=2, label=f"{label}")
        ax.fill_between(T_common, p10, p90, color=col, alpha=0.15)
    ax.set_yscale("log")
    ax.set_ylim(1e-4, np.percentile(q_pr.max(axis=1), 99) * 2)
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel(r"Tidal heating $\dot{q}$ ($\mu$W/m³)")
    ax.set_title(r"(b) $\dot{q}(T)$ — log scale, by thickness bin")
    ax.axvline(255, color='red', ls='--', lw=0.8, alpha=0.5, label=r"$T_{opt}$ (η≈μ/ω)")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.2, which="both")

    # ------------------------------------------------------------------
    # (c) Tidal heating q̇(z/H) — binned, LOG x-axis
    # ------------------------------------------------------------------
    ax = axes[2]
    for label, lo, hi, col in BINS:
        mask = (H >= lo) & (H < hi)
        zn, med, p10, p90 = _envelope(q_pr, H, mask)
        if med is None:
            continue
        ax.plot(med, zn, color=col, lw=2, label=f"{label}")
        ax.fill_betweenx(zn, np.maximum(p10, 1e-5), p90, color=col, alpha=0.15)
    ax.set_xscale("log")
    ax.set_xlim(1e-4, np.percentile(q_pr.max(axis=1), 99) * 2)
    ax.set_xlabel(r"Tidal heating $\dot{q}$ ($\mu$W/m³)")
    ax.set_ylabel("Normalised depth  z / H")
    ax.invert_yaxis()
    ax.set_title(r"(c) $\dot{q}(z/H)$ — log scale, by thickness bin")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.2, which="both")

    fig.suptitle(f"Diagnostic Heating Profiles — Subcritical {tag.title()} (N={n:,})",
                 fontsize=15, y=1.01)
    fig.tight_layout()
    outname = f"diagnostic_profiles_{tag}_subcrit.png"
    fig.savefig(os.path.join(FIGURES_DIR, outname), dpi=200, bbox_inches="tight")
    print(f"  Saved to figures/{outname}")


# ── viscosity figure ─────────────────────────────────────────────────────────
def plot_viscosity(datapath, tag="global"):
    """6-panel viscosity diagnostic figure."""
    import matplotlib.pyplot as plt

    d = np.load(datapath)
    # Mask for subcritical only
    mask_subcrit = d['Nu'] == 1.0
    
    H      = d['H_km'][mask_subcrit]
    T_pr   = d['T_profiles'][mask_subcrit]         # (n, nx)  K
    eta_pr = d['eta_profiles'][mask_subcrit]       # (n, nx)  Pa·s
    D_cond = d['D_cond_km'][mask_subcrit]
    D_conv = d['D_conv_km'][mask_subcrit]
    d_grain = d['d_grain'][mask_subcrit]           # (n,)  m
    Nu      = d['Nu'][mask_subcrit]
    n  = len(H)
    if n == 0:
        return
    nx = T_pr.shape[1]

    # Optimal viscosity for Maxwell peak
    mu_shear = 3.3e9   # Pa  (Rheology.RIGIDITY_ICE)
    omega    = 2.047e-5 # s⁻¹ (Planetary.ORBITAL_FREQ)
    eta_opt  = mu_shear / omega  # ≈ 1.6e14 Pa·s

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # ------------------------------------------------------------------
    # (a) η(z/H) — binned median + 10-90 % band, LOG x
    # ------------------------------------------------------------------
    ax = axes[0, 0]
    log_eta = np.log10(eta_pr)
    for label, lo, hi, col in BINS:
        mask = (H >= lo) & (H < hi)
        zn, med, p10, p90 = _envelope(log_eta, H, mask)
        if med is None:
            continue
        ax.plot(10**med, zn, color=col, lw=2, label=f"{label} (N={mask.sum()})")
        ax.fill_betweenx(zn, 10**p10, 10**p90, color=col, alpha=0.15)
    ax.axvline(eta_opt, color='red', ls='--', lw=1.2, alpha=0.7,
               label=r'$\eta_{opt}=\mu/\omega$')
    ax.set_xscale("log")
    ax.set_xlabel("Viscosity  η  (Pa·s)")
    ax.set_ylabel("Normalised depth  z / H")
    ax.invert_yaxis()
    ax.set_title("(a) Viscosity η(z/H) — by thickness bin")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.2, which="both")

    # ------------------------------------------------------------------
    # (b) η(T) — binned, LOG y
    # ------------------------------------------------------------------
    ax = axes[0, 1]
    T_common = np.linspace(50, 273, 300)
    for label, lo, hi, col in BINS:
        mask = (H >= lo) & (H < hi)
        stack = []
        for i in np.where(mask)[0]:
            order = np.argsort(T_pr[i])
            T_sorted = T_pr[i][order]
            eta_sorted = np.log10(eta_pr[i][order])
            stack.append(np.interp(T_common, T_sorted, eta_sorted,
                                   left=eta_sorted[0], right=eta_sorted[-1]))
        if len(stack) == 0:
            continue
        stack = np.array(stack)
        med = np.median(stack, axis=0)
        p10 = np.percentile(stack, 10, axis=0)
        p90 = np.percentile(stack, 90, axis=0)
        ax.plot(T_common, 10**med, color=col, lw=2, label=f"{label}")
        ax.fill_between(T_common, 10**p10, 10**p90, color=col, alpha=0.12)
    ax.axhline(eta_opt, color='red', ls='--', lw=1.2, alpha=0.7,
               label=r'$\eta_{opt}=\mu/\omega$')
    ax.set_yscale("log")
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("Viscosity  η  (Pa·s)")
    ax.set_title("(b) η(T) — log scale, by thickness bin")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.2, which="both")

    # ------------------------------------------------------------------
    # (c) Basal viscosity vs shell thickness
    # ------------------------------------------------------------------
    ax = axes[0, 2]
    eta_base = eta_pr[:, -1]    # viscosity at ice-ocean interface
    for label, lo, hi, col in BINS:
        mask = (H >= lo) & (H < hi)
        ax.scatter(H[mask], eta_base[mask], c=col, s=8, alpha=0.4,
                   edgecolors='none', label=label)
    ax.axhline(eta_opt, color='red', ls='--', lw=1.2, alpha=0.7,
               label=r'$\eta_{opt}$')
    ax.set_yscale("log")
    ax.set_xlabel("Total ice shell thickness (km)")
    ax.set_ylabel(r"Basal viscosity  $\eta_{base}$  (Pa·s)")
    ax.set_title("(c) Basal viscosity vs Thickness")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.2, which="both")

    # ------------------------------------------------------------------
    # (d) Fraction of shell at η < η_opt  (tidal heating "active zone")
    # ------------------------------------------------------------------
    ax = axes[1, 0]
    frac_active = np.array([np.mean(eta_pr[i] <= eta_opt) for i in range(n)])
    for label, lo, hi, col in BINS:
        mask = (H >= lo) & (H < hi)
        ax.scatter(H[mask], frac_active[mask] * 100, c=col, s=8, alpha=0.4,
                   edgecolors='none', label=label)
    # Running median
    sort_idx = np.argsort(H)
    H_s = H[sort_idx]
    fa_s = (frac_active * 100)[sort_idx]
    win = max(n // 30, 20)
    med_H = np.convolve(H_s, np.ones(win)/win, mode='valid')
    nmed = len(med_H)
    med_fa = np.array([np.median(fa_s[max(0,i-win//2):i+win//2])
                        for i in range(win//2, win//2 + nmed)])
    ax.plot(med_H, med_fa, 'k-', lw=2, label='Running median')
    ax.set_xlabel("Total ice shell thickness (km)\n[Note: Thicker shells in ensemble correlate with larger $d_{grain}$]")
    ax.set_ylabel(r"Shell fraction with $\eta \leq \eta_{opt}$  (%)")
    ax.set_title(r"(d) 'Active zone' fraction across MC ensemble")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.2)

    # ------------------------------------------------------------------
    # (e) η vs d_grain (coloured by thickness)
    # ------------------------------------------------------------------
    ax = axes[1, 1]
    # Use mid-shell viscosity as representative
    eta_mid = eta_pr[:, nx // 2]
    sc = ax.scatter(d_grain * 1e3, eta_mid, c=H, s=8, alpha=0.5,
                    cmap='viridis', edgecolors='none',
                    vmin=np.percentile(H, 2), vmax=np.percentile(H, 98))
    fig.colorbar(sc, ax=ax, label="H (km)")
    ax.axhline(eta_opt, color='red', ls='--', lw=1.2, alpha=0.7,
               label=r'$\eta_{opt}$')
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Grain size  d  (mm)")
    ax.set_ylabel(r"Mid-shell viscosity  $\eta_{mid}$  (Pa·s)")
    ax.set_title("(e) Viscosity vs Grain size")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.2, which="both")

    # ------------------------------------------------------------------
    # (f) Viscosity range within each shell (max/min)
    # ------------------------------------------------------------------
    ax = axes[1, 2]
    eta_min = eta_pr.min(axis=1)
    eta_max = eta_pr.max(axis=1)
    eta_range = np.log10(eta_max) - np.log10(eta_min)  # orders of magnitude
    for label, lo, hi, col in BINS:
        mask = (H >= lo) & (H < hi)
        ax.scatter(H[mask], eta_range[mask], c=col, s=8, alpha=0.4,
                   edgecolors='none', label=label)
    er_s = eta_range[sort_idx]
    med_er = np.array([np.median(er_s[max(0,i-win//2):i+win//2])
                        for i in range(win//2, win//2 + nmed)])
    ax.plot(med_H, med_er, 'k-', lw=2, label='Running median')
    ax.set_xlabel("Total ice shell thickness (km)")
    ax.set_ylabel("Viscosity range  (orders of magnitude)")
    ax.set_title("(f) Viscosity dynamic range vs Thickness")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.2)

    fig.suptitle(f"Viscosity Diagnostic — Subcritical {tag.title()} (N={n:,})",
                 fontsize=15, y=1.01)
    fig.tight_layout()
    outname = f"viscosity_profiles_{tag}_subcrit.png"
    fig.savefig(os.path.join(FIGURES_DIR, outname), dpi=200, bbox_inches="tight")
    print(f"  Saved to figures/{outname}")


# ═════════════════════════════════════════════════════════════════════════════
def plot_diagnostics(datapath, tag="global"):
    import matplotlib.pyplot as plt

    d = np.load(datapath)
    mask_subcrit = d["Nu"] == 1.0

    H = d["H_km"][mask_subcrit]
    T_pr = d["T_profiles"][mask_subcrit].copy()
    z_pr = d["z_profiles"][mask_subcrit]
    q_pr = d["q_profiles"][mask_subcrit] * 1e6
    D_cond = d["D_cond_km"][mask_subcrit]
    Nu = d["Nu"][mask_subcrit]
    n = len(H)
    if n == 0:
        print(f"No subcritical models found in {datapath}!")
        return

    if "T_c" in d:
        T_c = d["T_c"][mask_subcrit]
        for i in range(n):
            if Nu[i] > 1.0 and D_cond[i] < H[i]:
                is_conv = z_pr[i] > D_cond[i]
                if np.any(is_conv):
                    T_base = T_pr[i, -1]
                    T_pr[i, is_conv] = T_c[i]
                    T_pr[i, -1] = T_base

    fig, axes = plt.subplots(1, 3, figsize=figsize_double(0.40))

    ax = axes[0]
    for label, lo, hi, col in BINS:
        mask = (H >= lo) & (H < hi)
        zn, med, p10, p90 = _envelope(T_pr, H, mask)
        if med is None:
            continue
        ax.plot(med, zn, color=col, lw=1.4, label=f"{label} (N={mask.sum()})")
        ax.fill_betweenx(zn, p10, p90, color=col, alpha=0.12)
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("Normalised depth  $z / H$")
    ax.invert_yaxis()
    ax.legend(fontsize=5.8, loc="lower left")
    add_minor_gridlines(ax)
    label_panel(ax, "a")

    ax = axes[1]
    T_common = np.linspace(100, 273, 200)
    for label, lo, hi, col in BINS:
        mask = (H >= lo) & (H < hi)
        stack = []
        for i in np.where(mask)[0]:
            order = np.argsort(T_pr[i])
            stack.append(np.interp(T_common, T_pr[i][order], q_pr[i][order], left=0, right=0))
        if not stack:
            continue
        stack = np.array(stack)
        med = np.median(stack, axis=0)
        p10 = np.percentile(stack, 10, axis=0)
        p90 = np.percentile(stack, 90, axis=0)
        ax.plot(T_common, med, color=col, lw=1.4, label=label)
        ax.fill_between(T_common, p10, p90, color=col, alpha=0.10)
    ax.set_yscale("log")
    ax.set_ylim(1e-4, np.percentile(q_pr.max(axis=1), 99) * 2)
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel(r"Tidal heating $\dot{q}$ ($\mu$W m$^{-3}$)")
    ax.axvline(255, color="0.4", ls=":", lw=0.7, alpha=0.8)
    ax.text(256.5, ax.get_ylim()[1] * 0.42, r"$T_\mathrm{opt}$",
            fontsize=6.5, color="0.35", va="center")
    ax.legend(fontsize=5.8, loc="upper left")
    add_minor_gridlines(ax)
    label_panel(ax, "b")

    ax = axes[2]
    for label, lo, hi, col in BINS:
        mask = (H >= lo) & (H < hi)
        zn, med, p10, p90 = _envelope(q_pr, H, mask)
        if med is None:
            continue
        ax.plot(med, zn, color=col, lw=1.4, label=label)
        ax.fill_betweenx(zn, np.maximum(p10, 1e-5), p90, color=col, alpha=0.10)
    ax.set_xscale("log")
    ax.set_xlim(1e-4, np.percentile(q_pr.max(axis=1), 99) * 2)
    ax.set_xlabel(r"Tidal heating $\dot{q}$ ($\mu$W m$^{-3}$)")
    ax.set_ylabel("Normalised depth  $z / H$")
    ax.invert_yaxis()
    ax.legend(fontsize=5.8, loc="upper right")
    add_minor_gridlines(ax)
    label_panel(ax, "c")

    fig.suptitle(f"Diagnostic heating profiles - Subcritical {tag.title()} (N = {n:,})",
                 fontsize=9, y=1.03)
    fig.tight_layout(w_pad=1.8)
    save_fig(fig, f"diagnostic_profiles_{tag}_subcrit", FIGURES_DIR)


def plot_viscosity(datapath, tag="global"):
    import matplotlib.pyplot as plt

    d = np.load(datapath)
    mask_subcrit = d["Nu"] == 1.0

    H = d["H_km"][mask_subcrit]
    T_pr = d["T_profiles"][mask_subcrit]
    eta_pr = d["eta_profiles"][mask_subcrit]
    n = len(H)
    if n == 0:
        print(f"No subcritical models found in {datapath}!")
        return

    fig, axes = plt.subplots(1, 2, figsize=figsize_double(0.38))

    ax = axes[0]
    log_eta = np.log10(eta_pr)
    for label, lo, hi, col in BINS:
        mask = (H >= lo) & (H < hi)
        zn, med, p10, p90 = _envelope(log_eta, H, mask)
        if med is None:
            continue
        ax.plot(10**med, zn, color=col, lw=1.4, label=f"{label} (N={mask.sum()})")
        ax.fill_betweenx(zn, 10**p10, 10**p90, color=col, alpha=0.12)
    ax.set_xscale("log")
    ax.set_xlabel("Viscosity $\\eta$ (Pa s)")
    ax.set_ylabel("Normalised depth  $z / H$")
    ax.invert_yaxis()
    ax.legend(fontsize=5.8, loc="upper left")
    add_minor_gridlines(ax)
    label_panel(ax, "a")

    ax = axes[1]
    T_common = np.linspace(50, 273, 300)
    for label, lo, hi, col in BINS:
        mask = (H >= lo) & (H < hi)
        stack = []
        for i in np.where(mask)[0]:
            order = np.argsort(T_pr[i])
            eta_sorted = np.log10(eta_pr[i][order])
            stack.append(np.interp(T_common, T_pr[i][order], eta_sorted,
                                   left=eta_sorted[0], right=eta_sorted[-1]))
        if not stack:
            continue
        stack = np.array(stack)
        med = np.median(stack, axis=0)
        p10 = np.percentile(stack, 10, axis=0)
        p90 = np.percentile(stack, 90, axis=0)
        ax.plot(T_common, 10**med, color=col, lw=1.4, label=label)
        ax.fill_between(T_common, 10**p10, 10**p90, color=col, alpha=0.12)
    ax.set_yscale("log")
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("Viscosity $\\eta$ (Pa s)")
    ax.legend(fontsize=5.8, loc="upper right")
    add_minor_gridlines(ax)
    label_panel(ax, "b")

    fig.suptitle(f"Viscosity diagnostics - Subcritical {tag.title()} (N = {n:,})",
                 fontsize=9, y=1.03)
    fig.tight_layout(w_pad=1.6)
    save_fig(fig, f"viscosity_profiles_{tag}_subcrit", FIGURES_DIR)


if __name__ == "__main__":
    mp.freeze_support()

    # Parse which scenario to run: global (default), equator, pole
    SCENARIOS = {
        "global":  (UpdatedPtidalSampler,      "diagnostic_profiles.npz",          "global",  10000),
        "equator": (UpdatedEquatorSampler,     "diagnostic_profiles_equator.npz",  "equator", 10000),
        "pole":    (UpdatedPoleSampler,        "diagnostic_profiles_pole.npz",     "pole",    10000),
    }

    scenario = "global"
    for arg in sys.argv[1:]:
        if arg in SCENARIOS:
            scenario = arg

    sampler_cls, npz_name, tag, default_n = SCENARIOS[scenario]
    outpath = os.path.join(RESULTS_DIR, npz_name)

    if os.path.exists(outpath) and "--run" not in sys.argv:
        print(f"Using existing {scenario} results (pass --run to re-compute)")
    else:
        valid = run_diagnostics(n_samples=default_n, seed=44,
                                sampler_class=sampler_cls)
        save_diagnostics(valid, outpath)

    plot_diagnostics(outpath, tag=tag)
    plot_viscosity(outpath, tag=tag)
