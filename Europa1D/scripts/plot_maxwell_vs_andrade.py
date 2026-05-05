"""
Maxwell vs Andrade tidal heating diagnostic comparison.

Computes tidal dissipation profiles through the ice shell for both rheological
models using the same viscosity structure, illustrating:
  (a) q_tidal(T) — the resonance structure and how Andrade broadens the peak
  (b) q_tidal(η) — dissipation as function of viscosity (the mechanical response)
  (c) q_tidal(z/H) — depth profiles for a representative 25 km convecting shell
  (d) Andrade/Maxwell ratio — where Andrade dominates

References:
  Maxwell:  q = ε₀²ω²η / [2(1 + (ωη/μ)²)]
  Andrade:  q = ½ ω ε₀² Im(G*), with complex compliance from McCarthy et al. (2011)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from scipy.special import gamma
import matplotlib.pyplot as plt
import matplotlib as mpl

# ── Import project physics ──────────────────────────────────────────────────
from constants import Thermal, Planetary, Rheology, HeatFlux
from Physics import IcePhysics

# ── Pub style (inline for self-contained script) ───────────────────────────
SINGLE_COL = 3.50
DOUBLE_COL = 7.20

class PAL:
    BLUE   = "#0072B2"
    ORANGE = "#E69F00"
    GREEN  = "#009E73"
    RED    = "#D55E00"
    PURPLE = "#CC79A7"
    CYAN   = "#56B4E9"

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8,
    "mathtext.fontset": "stixsans",
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "axes.titleweight": "bold",
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "lines.linewidth": 1.4,
    "legend.fontsize": 7,
    "legend.frameon": False,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
})

FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)


# ── Tidal heating functions (bypass config toggle, compute both explicitly) ─
def tidal_maxwell(eta, strain, omega, mu):
    """Maxwell viscoelastic dissipation [W/m³]."""
    num = strain**2 * omega**2 * eta
    den = 2.0 * (1.0 + (omega * eta / mu)**2)
    return num / den


def tidal_andrade(eta, strain, omega, mu, alpha=0.2, zeta=1.0):
    """Andrade transient creep dissipation [W/m³] (McCarthy et al. 2011)."""
    J_elastic = 1.0 / mu
    tau = eta / mu
    andrade_term = np.clip(omega * tau * zeta, 1e-100, None)
    const_term = J_elastic * andrade_term**(-alpha) * gamma(1 + alpha)

    J_real = J_elastic + const_term * np.cos(alpha * np.pi / 2.0)
    J_imag = J_elastic / (omega * tau) + const_term * np.sin(alpha * np.pi / 2.0)
    G_imag = J_imag / (J_real**2 + J_imag**2)
    return 0.5 * omega * strain**2 * G_imag


# ── Physical parameters ────────────────────────────────────────────────────
strain = HeatFlux.TIDAL_STRAIN        # 1e-5
omega  = Planetary.ORBITAL_FREQ       # 2.047e-5 s⁻¹
mu     = Rheology.RIGIDITY_ICE        # 3.3e9 Pa
eta_opt = mu / omega                  # ~1.6e14 Pa·s  (Maxwell resonance)

# Temperature range spanning surface to base
T = np.linspace(90, 273, 2000)

# Composite viscosity at each temperature (Howell 2021 defaults)
eta = IcePhysics.composite_viscosity(T)

# Compute heating for both models
q_maxwell = tidal_maxwell(eta, strain, omega, mu)
q_andrade = tidal_andrade(eta, strain, omega, mu)

# ── Representative depth profile (25 km convecting shell) ─────────────────
H = 25e3  # m
T_surf = 104.0  # K
T_base = 273.0  # K
nz = 500
z_norm = np.linspace(0, 1, nz)  # z/H

# Conductive-lid temperature profile with isothermal convective core
D_cond_frac = 0.45  # typical lid fraction
T_c = 250.0         # convective transition temperature

T_profile = np.where(
    z_norm < D_cond_frac,
    T_surf + (T_c - T_surf) * (z_norm / D_cond_frac),
    np.where(z_norm < 0.98, T_c, T_c + (T_base - T_c) * (z_norm - 0.98) / 0.02),
)

eta_profile = IcePhysics.composite_viscosity(T_profile)
q_m_profile = tidal_maxwell(eta_profile, strain, omega, mu)
q_a_profile = tidal_andrade(eta_profile, strain, omega, mu)


# ============================================================================
#  FIGURE
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL, DOUBLE_COL * 0.72))

# ── (a) q_tidal vs Temperature ─────────────────────────────────────────────
ax = axes[0, 0]
ax.semilogy(T, q_maxwell * 1e6, color=PAL.BLUE, lw=1.6, label="Maxwell")
ax.semilogy(T, q_andrade * 1e6, color=PAL.RED,  lw=1.6, label="Andrade (α=0.2)")
ax.axvline(T[np.argmax(q_maxwell)], color=PAL.BLUE, ls=':', lw=0.7, alpha=0.5)
ax.axvline(T[np.argmax(q_andrade)], color=PAL.RED,  ls=':', lw=0.7, alpha=0.5)
ax.set_xlabel("Temperature (K)")
ax.set_ylabel(r"$\dot{q}_{\rm tidal}$ ($\mu$W m$^{-3}$)")
ax.set_ylim(1e-6, None)
ax.legend(loc="upper left")
ax.set_title("(a)  Dissipation vs temperature")
ax.grid(True, which="both", alpha=0.15, lw=0.3)
ax.minorticks_on()

# ── (b) q_tidal vs Viscosity ──────────────────────────────────────────────
ax = axes[0, 1]
ax.loglog(eta, q_maxwell * 1e6, color=PAL.BLUE, lw=1.6, label="Maxwell")
ax.loglog(eta, q_andrade * 1e6, color=PAL.RED,  lw=1.6, label="Andrade (α=0.2)")
ax.axvline(eta_opt, color='0.4', ls='--', lw=0.8, alpha=0.7,
           label=r"$\eta_{\rm opt} = \mu/\omega$")
ax.set_xlabel(r"Viscosity $\eta$ (Pa$\cdot$s)")
ax.set_ylabel(r"$\dot{q}_{\rm tidal}$ ($\mu$W m$^{-3}$)")
ax.set_ylim(1e-6, None)
ax.legend(loc="lower left", fontsize=6.5)
ax.set_title(r"(b)  Dissipation vs viscosity")
ax.grid(True, which="both", alpha=0.15, lw=0.3)

# ── (c) Depth profiles ────────────────────────────────────────────────────
ax = axes[1, 0]
ax.semilogx(q_m_profile * 1e6, z_norm, color=PAL.BLUE, lw=1.6, label="Maxwell")
ax.semilogx(q_a_profile * 1e6, z_norm, color=PAL.RED,  lw=1.6, label="Andrade")
ax.axhline(D_cond_frac, color='0.4', ls='--', lw=0.7, alpha=0.6)
ax.text(ax.get_xlim()[0] * 5, D_cond_frac - 0.03, "lid base",
        fontsize=6.5, color='0.4', va='bottom')
ax.set_xlabel(r"$\dot{q}_{\rm tidal}$ ($\mu$W m$^{-3}$)")
ax.set_ylabel("Normalised depth  $z / H$")
ax.invert_yaxis()
ax.legend(loc="lower right")
ax.set_title("(c)  Depth profile (H = 25 km)")
ax.grid(True, which="both", alpha=0.15, lw=0.3)
ax.minorticks_on()

# ── (d) Andrade / Maxwell ratio ───────────────────────────────────────────
ax = axes[1, 1]
ratio = q_andrade / np.maximum(q_maxwell, 1e-30)
ax.semilogy(T, ratio, color=PAL.PURPLE, lw=1.6)
ax.axhline(1.0, color='0.5', ls='-', lw=0.5)
ax.fill_between(T, 1, ratio, where=ratio > 1,
                color=PAL.PURPLE, alpha=0.08, label="Andrade > Maxwell")
ax.fill_between(T, ratio, 1, where=ratio < 1,
                color=PAL.CYAN, alpha=0.08, label="Maxwell > Andrade")
ax.set_xlabel("Temperature (K)")
ax.set_ylabel("Andrade / Maxwell ratio")
ax.set_title("(d)  Rheology ratio")
ax.legend(loc="upper left", fontsize=6.5)
ax.grid(True, which="both", alpha=0.15, lw=0.3)
ax.minorticks_on()

fig.suptitle("Maxwell vs Andrade Tidal Dissipation — Diagnostic Comparison",
             fontsize=10, fontweight="bold", y=1.01)
fig.tight_layout()

outpath = os.path.join(FIGURES_DIR, "maxwell_vs_andrade_comparison.png")
fig.savefig(outpath, dpi=300, bbox_inches="tight")
print(f"Saved: {outpath}")

outpath_pdf = os.path.join(FIGURES_DIR, "maxwell_vs_andrade_comparison.pdf")
fig.savefig(outpath_pdf, transparent=True)
print(f"Saved: {outpath_pdf}")
plt.show()
