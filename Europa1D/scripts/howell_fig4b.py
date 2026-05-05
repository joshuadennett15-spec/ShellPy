import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from scipy.stats import truncnorm

FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')


def cbe_analysis(data: np.ndarray) -> dict:
    counts, bins = np.histogram(data, bins=1000, density=True)
    smoothed = savgol_filter(counts, 51, 3)
    smoothed[smoothed < 0] = 0
    cbe_idx = np.argmax(smoothed)
    cbe = bins[cbe_idx]

    n_cum = np.cumsum(counts)
    n_cum = n_cum / n_cum.max()
    cbe_cum = n_cum[cbe_idx]
    cbe_cum_lo = cbe_cum * 0.341
    cbe_cum_hi = (1 - cbe_cum) * 0.341

    cbe_lo = cbe - bins[(np.abs(n_cum - (cbe_cum - cbe_cum_lo))).argmin()]
    cbe_hi = bins[(np.abs(n_cum - (cbe_cum + cbe_cum_hi))).argmin()] - cbe

    return {
        "cbe": cbe,
        "lo": cbe_lo,
        "hi": cbe_hi,
        "bins": bins[:-1],
        "pdf": smoothed / smoothed.max(),
        "cdf": n_cum,
    }


def cond_analysis(values: np.ndarray,
                  thickness_samples: np.ndarray,
                  thickness_cbe: float,
                  dice: np.ndarray) -> np.ndarray:
    lo = thickness_samples[thickness_samples <= thickness_cbe]
    hi = thickness_samples[thickness_samples > thickness_cbe]
    lo_cbe = cbe_analysis(lo)
    hi_cbe = cbe_analysis(hi)

    f_keep = interp1d(
        np.append(lo_cbe["bins"], hi_cbe["bins"]),
        np.append(lo_cbe["cdf"], 1 - hi_cbe["cdf"]),
        fill_value=(0, 0),
        bounds_error=False,
    )
    p_keep = f_keep(thickness_samples)
    keep = dice <= p_keep
    return values[keep]


def _filter(mask: np.ndarray, *arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    return tuple(arr[mask] for arr in arrays)


def main(n_samples: int = 200_000, seed: int = 42) -> None:
    rng = np.random.default_rng(seed)

    # Constants (Howell 2021)
    R = 8.31
    g = 1.315
    r_E = 1561e3
    SA_Europa = 4 * np.pi * r_E ** 2
    omega = 2.047e-5
    T_m0 = 273.0
    rho0 = 917.0
    Cp = 2000.0
    alpha = 1.6e-4
    d_Vm = 1.97e-5

    # Step 1: Interior heat flux
    T_s = rng.normal(104.0, 7.0, n_samples)
    D_H2O = rng.normal(127e3, 21e3, n_samples)

    Q_ir = rng.normal(4.5e-12, 1.0e-12, n_samples)
    mask = Q_ir > 0
    T_s, D_H2O, Q_ir = _filter(mask, T_s, D_H2O, Q_ir)
    n_samples = len(Q_ir)

    q_ir = Q_ir * (4.80e22) / SA_Europa
    Pi = 10 ** rng.normal(np.log10(100e9), 1 / 3, n_samples)
    q_it = Pi / SA_Europa
    q_i = q_ir + q_it

    # Step 2: Conductive layer properties
    f_s = 10 ** rng.normal(np.log10(0.03), 1 / 3, n_samples)
    f_s = np.clip(f_s, 0, 0.22)
    T_m = T_m0 - 21 * f_s / 0.22

    phi = rng.uniform(0.0, 0.3, n_samples)
    T_phi = rng.normal(150.0, 20.0 / 3.0, n_samples)

    # Diffusion parameters
    D0v = rng.normal(9.1e-4, 0.033 * 9.1e-4, n_samples)
    Qv = rng.normal(59.4e3, 0.05 * 59.4e3, n_samples)
    D0b = rng.normal(8.4e-4, 0.033 * 8.4e-4, n_samples)
    Qb = rng.normal(49.0e3, 0.05 * 49.0e3, n_samples)

    T_c = (np.sqrt(4 * T_m * (R / Qv) + 1) - 1) / (2 * (R / Qv))
    T_cond_base = 2 * T_c - T_m

    mask = (T_phi > T_s) & (T_phi < T_m) & (T_c > T_phi) & (T_c < T_m)
    T_s, D_H2O, q_i, T_m, T_c, T_cond_base, phi, f_s, D0v, D0b, Qv, Qb, T_phi = _filter(
        mask, T_s, D_H2O, q_i, T_m, T_c, T_cond_base, phi, f_s, D0v, D0b, Qv, Qb, T_phi
    )
    n_samples = len(T_s)

    f_phi = np.log(T_phi / T_s) / np.log(T_cond_base / T_s)
    T_T = (T_s + T_cond_base) / 2
    k_T = 567 / T_T
    T_pore = (T_s + T_phi) / 2
    k_pore = 567 / T_pore
    T_solid = (T_phi + T_cond_base) / 2
    k_solid = 567 / T_solid
    k_Tp = (f_phi * k_pore * (1 - phi)) + ((1 - f_phi) * k_solid)

    B_k = 10 ** rng.uniform(-1, 1, n_samples)
    k_Tps = ((1 - f_s) * k_Tp) + (f_s * k_Tp * B_k)
    K = k_Tps / (rho0 * Cp)

    # Step 3: Conductive thickness bounds
    Dv = D0v * np.exp(-Qv / (R * T_c))
    Db = D0b * np.exp(-Qb / (R * T_c))

    d = 10 ** rng.normal(np.log10(1e-3), 1.0, n_samples)
    d_del = rng.normal(np.mean([9.04e-10, 5.22e-10]),
                       np.std([9.04e-10, 5.22e-10]), n_samples)

    eta_c = 0.5 * ((42 * d_Vm / (R * T_c * d ** 2)) * (Dv + (np.pi * d_del / d) * Db)) ** -1
    mask = (np.log10(eta_c) >= 10) & (np.log10(eta_c) <= 25)
    T_s, D_H2O, q_i, T_m, T_c, T_cond_base, phi, f_s, k_Tps, K, eta_c = _filter(
        mask, T_s, D_H2O, q_i, T_m, T_c, T_cond_base, phi, f_s, k_Tps, K, eta_c
    )
    n_samples = len(T_s)

    G_mean = 3.5e9
    G_sigma = 0.5e9
    G_conv = G_mean * truncnorm.rvs(a=-np.inf, b=0, loc=1, scale=(G_sigma / G_mean), size=n_samples, random_state=rng)
    G_conv = np.clip(G_conv, G_mean / 20, None)

    epsilon = rng.normal(1e-5, 0.05e-5, n_samples)
    dqc_dz = (epsilon ** 2 * omega ** 2 * eta_c) / (1 + omega ** 2 * eta_c ** 2 / G_conv ** 2)

    D_conv_min = (567 / T_c) * (T_m - T_cond_base) / q_i
    q_conv_min = dqc_dz * D_conv_min

    Ra_crit = 1e6
    D_max_Ra = ((Ra_crit * K * eta_c) / (alpha * (T_cond_base - T_s) * rho0 * g)) ** (1 / 3)
    D_max_q_i = k_Tps * (T_cond_base - T_s) / (q_i + q_conv_min)
    D_max_Ra[np.isnan(D_max_Ra)] = np.inf
    D_max = np.minimum(D_max_q_i, D_max_Ra)

    q_c_max = D_H2O[:n_samples] * dqc_dz
    D_min = k_Tps * (T_cond_base - T_s) / (q_c_max + q_i)

    # Step 4: Sample layers
    mask = D_min < D_max
    D_min, D_max, T_s, D_H2O, q_i, T_cond_base, k_Tps, dqc_dz = _filter(
        mask, D_min, D_max, T_s, D_H2O, q_i, T_cond_base, k_Tps, dqc_dz
    )
    n_samples = len(D_min)

    D_cond = rng.uniform(D_min, D_max)
    q_c = k_Tps * (T_cond_base - T_s) / D_cond - q_i
    D_conv = q_c / dqc_dz
    D_tot = D_cond + D_conv

    valid = (D_min < D_max) & (D_cond > 0) & (D_conv > 0) & (D_tot > 0)
    valid &= (D_cond < D_H2O[:n_samples]) & (D_conv < D_H2O[:n_samples]) & (D_tot < D_H2O[:n_samples])
    D_cond, D_conv, D_tot = _filter(valid, D_cond, D_conv, D_tot)

    # Conditional distributions
    dice = rng.random(len(D_tot)) / 2 + 0.5
    D_tot_cbe = cbe_analysis(D_tot)
    D_cond_cond = cond_analysis(D_cond, D_tot, D_tot_cbe["cbe"], dice)
    D_conv_cond = cond_analysis(D_conv, D_tot, D_tot_cbe["cbe"], dice)

    D_cond_cbe = cbe_analysis(D_cond_cond)
    D_conv_cbe = cbe_analysis(D_conv_cond)

    # Plot
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    conductive_color = "#17BECF"
    convective_color = "#C44E52"

    cond_x = D_cond_cbe["bins"] / 1e3
    cond_y = D_cond_cbe["pdf"] * 0.9
    conv_x = D_conv_cbe["bins"] / 1e3
    conv_y = D_conv_cbe["pdf"] * 0.9

    ax.plot(cond_x, cond_y, color=conductive_color, lw=2.0)
    ax.plot(conv_x, conv_y, color=convective_color, lw=2.0)

    ax.set_xlabel("Layer Thickness [km]")
    ax.set_ylabel("Discrete Probability Density")
    ax.set_xlim(0, 60)
    y_max = max(cond_y.max(), conv_y.max()) if len(cond_y) and len(conv_y) else 1.0
    ax.set_ylim(0, y_max * 1.15)

    cond_peak_idx = int(np.argmax(cond_y)) if len(cond_y) else 0
    conv_peak_idx = int(np.argmax(conv_y)) if len(conv_y) else 0

    ax.text(
        conv_x[conv_peak_idx] + 1.5,
        min(y_max * 1.02, conv_y[conv_peak_idx] * 1.05),
        f"{D_conv_cbe['cbe']/1e3:.1f} +{D_conv_cbe['hi']/1e3:.1f} -{D_conv_cbe['lo']/1e3:.1f} km",
        color=convective_color,
        fontsize=8,
        ha="left",
        va="bottom",
    )
    ax.text(
        cond_x[cond_peak_idx] + 6,
        min(y_max * 0.95, cond_y[cond_peak_idx] * 1.1),
        f"{D_cond_cbe['cbe']/1e3:.1f} +{D_cond_cbe['hi']/1e3:.1f} -{D_cond_cbe['lo']/1e3:.1f} km",
        color=conductive_color,
        fontsize=8,
        ha="left",
        va="bottom",
    )

    ax.text(conv_x[conv_peak_idx] + 2, conv_y[conv_peak_idx] * 0.7, "Convective",
            color=convective_color, fontsize=9, rotation=-70)
    ax.text(cond_x[cond_peak_idx] + 12, cond_y[cond_peak_idx] * 0.5, "Conductive",
            color=conductive_color, fontsize=9, rotation=-70)

    ax.text(36, y_max * 0.5, f"{D_tot_cbe['cbe']/1e3:.1f} km", color=conductive_color, fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "howell_figure4b_like.png"), dpi=300)


if __name__ == "__main__":
    main()
