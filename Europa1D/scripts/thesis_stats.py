"""
Statistical comparison of Europa MC ensembles for thesis chapter.

Spec: docs/superpowers/specs/2026-03-19-mc-statistical-comparison-design.md
"""
from __future__ import annotations

import csv
import json
import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
from scipy import stats
from scipy.signal import savgol_filter
from statsmodels.stats.multitest import multipletests
from statsmodels.regression.quantile_regression import QuantReg

logger = logging.getLogger(__name__)

# NPZ key -> internal name
_QOI_KEY_MAP = {
    "thicknesses_km": "thickness_km",
    "D_cond_km": "D_cond_km",
    "D_conv_km": "D_conv_km",
    "lid_fractions": "lid_fraction",
    "Ra_values": "Ra",
    "Nu_values": "Nu",
}

# Whole-population QoIs for Block 2 pairwise tests
WHOLE_POP_QOIS = ("thickness_km", "D_cond_km", "lid_fraction")

# QoIs only meaningful for convective subpopulation
CONV_ONLY_QOIS = ("D_conv_km", "Ra", "Nu")

# All QoIs
ALL_QOIS = WHOLE_POP_QOIS + CONV_ONLY_QOIS

# JT-eligible QoIs (continuous, no heavy boundary mass)
JT_QOIS = ("thickness_km", "D_cond_km", "lid_fraction")

# Conductive threshold (see spec for rationale)
LID_FRACTION_COND_THRESHOLD = 0.999

# Default scenario paths relative to results/
DEFAULT_SCENARIOS = {
    "Global Audited": "mc_15000_optionA_v2_andrade.npz",
    "Eq Baseline": "eq_baseline_andrade.npz",
    "Eq Moderate": "eq_moderate_andrade.npz",
    "Eq Strong": "eq_strong_andrade.npz",
}

# Cap for consistent cross-scenario comparison
CONVERGENCE_SCHEDULE = [500, 1000, 2000, 5000, 10000, 14997]


def load_scenario(path: str) -> Dict[str, Any]:
    """Load an NPZ archive and return standardized qois, params, metadata."""
    d = np.load(path, allow_pickle=True)

    qois = {}
    for npz_key, internal_name in _QOI_KEY_MAP.items():
        if npz_key in d:
            qois[internal_name] = np.asarray(d[npz_key], dtype=float)

    # Auto-detect and filter zero-variance parameters
    params = {}
    excluded = []
    for key in sorted(d.keys()):
        if not key.startswith("param_"):
            continue
        arr = np.asarray(d[key], dtype=float)
        name = key[len("param_"):]
        mean_abs = max(abs(np.mean(arr)), 1e-30)
        rel_std = np.std(arr) / mean_abs
        if rel_std < 1e-6:
            excluded.append(name)
            continue
        params[name] = arr

    if excluded:
        logger.info("Excluded zero-variance params from %s: %s", path, excluded)

    metadata = {
        "path": str(path),
        "n_valid": len(next(iter(qois.values()))),
        "excluded_params": excluded,
    }
    # Route eq_enhancement to metadata if present
    if "param_eq_enhancement" in d:
        metadata["eq_enhancement"] = float(d["param_eq_enhancement"][0])

    return {"qois": qois, "params": params, "metadata": metadata}


def _cliffs_delta_from_u(U: float, n1: int, n2: int) -> float:
    """Cliff's delta derived from Mann-Whitney U. Range [-1, 1]."""
    return (2.0 * U) / (n1 * n2) - 1.0


def _jonckheere_terpstra(
    groups: Sequence[np.ndarray], alternative: str = "two-sided"
) -> tuple:
    """
    Jonckheere-Terpstra test for ordered alternatives.

    Tie-corrected asymptotic normal approximation (Hollander & Wolfe 1999).
    Returns (J_statistic, p_value).
    """
    k = len(groups)
    ns = [len(g) for g in groups]
    N = sum(ns)

    # J = sum of U_{ij} for i < j
    J = 0.0
    for i in range(k - 1):
        for j in range(i + 1, k):
            u_stat, _ = stats.mannwhitneyu(
                groups[i], groups[j], alternative="two-sided"
            )
            J += u_stat

    # Expected value under H0
    N_sq_sum = sum(n * n for n in ns)
    E_J = (N * N - N_sq_sum) / 4.0

    # Tie-corrected variance (Hollander & Wolfe 1999, eq. 6.18)
    all_values = np.concatenate(groups)
    _, tie_counts = np.unique(all_values, return_counts=True)

    # Terms for variance
    A = (N * (N - 1) * (2 * N + 5)
         - sum(n * (n - 1) * (2 * n + 5) for n in ns)
         - sum(int(t) * (int(t) - 1) * (2 * int(t) + 5) for t in tie_counts))

    B = (sum(n * (n - 1) * (n - 2) for n in ns)
         * sum(int(t) * (int(t) - 1) * (int(t) - 2) for t in tie_counts))

    C = (sum(n * (n - 1) for n in ns)
         * sum(int(t) * (int(t) - 1) for t in tie_counts))

    Var_J = A / 72.0 + B / (36.0 * N * (N - 1) * (N - 2)) + C / (8.0 * N * (N - 1))
    Var_J = max(Var_J, 1e-30)

    z = (J - E_J) / np.sqrt(Var_J)

    # When groups are stochastically increasing (group[0] < group[1] < ...),
    # mannwhitneyu(groups[i], groups[j]) for i<j returns a small U (group[i] loses),
    # so J < E_J and z < 0.  "increasing" therefore rejects in the left tail.
    if alternative == "increasing":
        p_value = stats.norm.cdf(z)
    elif alternative == "decreasing":
        p_value = 1.0 - stats.norm.cdf(z)
    else:
        p_value = 2.0 * (1.0 - stats.norm.cdf(abs(z)))

    return float(J), float(p_value)


def _kendall_w(rankings: np.ndarray) -> tuple:
    """
    Kendall's W (coefficient of concordance).

    rankings: (k_raters, n_items) array of ranks.
    Returns (W, chi2, p_value).
    """
    k, n = rankings.shape
    rank_sums = rankings.sum(axis=0)
    mean_rank_sum = rank_sums.mean()
    SS = float(np.sum((rank_sums - mean_rank_sum) ** 2))
    W = (12.0 * SS) / (k * k * (n ** 3 - n))
    chi2 = k * (n - 1) * W
    p_value = 1.0 - stats.chi2.cdf(chi2, df=n - 1)
    return float(W), float(chi2), float(p_value)


def _bca_ci(data: np.ndarray, stat_func, n_boot: int = 2000,
            alpha: float = 0.05, rng=None) -> tuple:
    """BCa bootstrap confidence interval."""
    if rng is None:
        rng = np.random.default_rng(0)
    n = len(data)
    observed = stat_func(data)

    boot_stats = np.array([
        stat_func(data[rng.integers(0, n, n)]) for _ in range(n_boot)
    ])

    # Bias correction (clip to avoid -inf/+inf at edges)
    prop = np.clip(np.mean(boot_stats < observed), 0.5 / n_boot, 1.0 - 0.5 / n_boot)
    z0 = stats.norm.ppf(prop)

    # Acceleration (jackknife, subsampled for large n to avoid O(n^2))
    jack_n = min(n, 2000)
    jack_idx = rng.choice(n, jack_n, replace=False) if jack_n < n else np.arange(n)
    jackknife = np.array([stat_func(np.delete(data, i)) for i in jack_idx])
    jack_mean = jackknife.mean()
    num = np.sum((jack_mean - jackknife) ** 3)
    den = 6.0 * (np.sum((jack_mean - jackknife) ** 2) ** 1.5)
    a = num / den if den != 0 else 0.0

    # Adjusted percentiles
    z_alpha = stats.norm.ppf(alpha / 2.0)
    z_1alpha = stats.norm.ppf(1.0 - alpha / 2.0)
    p_low = stats.norm.cdf(z0 + (z0 + z_alpha) / (1.0 - a * (z0 + z_alpha)))
    p_high = stats.norm.cdf(z0 + (z0 + z_1alpha) / (1.0 - a * (z0 + z_1alpha)))

    p_low = np.clip(p_low, 0.5 / n_boot, 1.0 - 0.5 / n_boot)
    p_high = np.clip(p_high, 0.5 / n_boot, 1.0 - 0.5 / n_boot)

    ci_low = float(np.percentile(boot_stats, 100 * p_low))
    ci_high = float(np.percentile(boot_stats, 100 * p_high))
    return ci_low, ci_high


def _cbe_savgol(data: np.ndarray) -> float:
    """Current Best Estimate = mode of Savitzky-Golay smoothed PDF."""
    if len(data) < 50:
        return float(np.median(data))
    n_bins = min(80, max(20, len(data) // 200))
    counts, edges = np.histogram(data, bins=n_bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    window = min(len(counts) - 1, 7)
    if window % 2 == 0:
        window -= 1
    if window < 3:
        return float(centers[np.argmax(counts)])
    smoothed = savgol_filter(counts, window, 2)
    return float(centers[np.argmax(smoothed)])


def descriptive_summary(data: Mapping[str, Any],
                        n_boot: int = 2000) -> Dict[str, Any]:
    """Block 1: descriptive statistics for all QoIs in data."""
    rng = np.random.default_rng(0)
    result: Dict[str, Any] = {}

    for name, arr in data["qois"].items():
        finite = arr[np.isfinite(arr)]
        if len(finite) == 0:
            continue
        ci_median = _bca_ci(finite, np.median, n_boot, rng=rng)
        ci_mean = _bca_ci(finite, np.mean, n_boot, rng=rng)
        entry = {
            "mean": float(np.mean(finite)),
            "median": float(np.median(finite)),
            "cbe": _cbe_savgol(finite),
            "std": float(np.std(finite, ddof=1)),
            "P5": float(np.percentile(finite, 5)),
            "P16": float(np.percentile(finite, 16)),
            "P25": float(np.percentile(finite, 25)),
            "P50": float(np.percentile(finite, 50)),
            "P75": float(np.percentile(finite, 75)),
            "P84": float(np.percentile(finite, 84)),
            "P95": float(np.percentile(finite, 95)),
            "IQR": float(np.percentile(finite, 75) - np.percentile(finite, 25)),
            "skewness": float(stats.skew(finite)),
            "kurtosis": float(stats.kurtosis(finite)),
            "ci_median_low": ci_median[0],
            "ci_median_high": ci_median[1],
            "ci_mean_low": ci_mean[0],
            "ci_mean_high": ci_mean[1],
            "n": len(finite),
        }
        # log10 stats for Ra and Nu
        if name in ("Ra", "Nu"):
            log_arr = np.log10(finite[finite > 0])
            if len(log_arr) > 0:
                entry["log10_mean"] = float(np.mean(log_arr))
                entry["log10_median"] = float(np.median(log_arr))
                entry["log10_std"] = float(np.std(log_arr, ddof=1))
        result[name] = entry

    # Conductive fraction
    if "lid_fraction" in data["qois"]:
        lf = data["qois"]["lid_fraction"]
        cond_frac = float(np.mean(lf >= LID_FRACTION_COND_THRESHOLD))
        ci = _bca_ci(
            (lf >= LID_FRACTION_COND_THRESHOLD).astype(float),
            np.mean, n_boot, rng=rng
        )
        result["conductive_fraction"] = cond_frac
        result["conductive_fraction_ci"] = {"low": ci[0], "high": ci[1]}

    return result


def pairwise_comparison(
    data_a: Mapping[str, Any],
    data_b: Mapping[str, Any],
    qois: Sequence[str],
) -> Dict[str, Any]:
    """Block 2: KS, Mann-Whitney, Cliff's delta, Cohen's d for each QoI."""
    result = {}
    for qoi in qois:
        a = data_a["qois"][qoi]
        b = data_b["qois"][qoi]
        a = a[np.isfinite(a)]
        b = b[np.isfinite(b)]

        ks_D, ks_p = stats.ks_2samp(a, b)
        u_stat, u_p = stats.mannwhitneyu(a, b, alternative="two-sided")
        n1, n2 = len(a), len(b)
        # Note: r_rb = 1 - 2U/(n1*n2) and cliff_d = 2U/(n1*n2) - 1,
        # so r_rb = -cliff_d by definition. Both are reported; cliff_d > 0
        # means sample A tends to exceed sample B.
        r_rb = 1.0 - (2.0 * u_stat) / (n1 * n2)
        cliff_d = _cliffs_delta_from_u(u_stat, n1, n2)
        pooled_std = np.sqrt(
            (np.var(a, ddof=1) * (n1 - 1) + np.var(b, ddof=1) * (n2 - 1))
            / (n1 + n2 - 2)
        )
        cohen_d = (np.mean(a) - np.mean(b)) / pooled_std if pooled_std > 0 else 0.0

        result[qoi] = {
            "ks_D": float(ks_D),
            "ks_p": float(ks_p),
            "mw_U": float(u_stat),
            "mw_p": float(u_p),
            "r_rb": float(r_rb),
            "cliff_d": float(cliff_d),
            "cohen_d": float(cohen_d),
            "n_a": n1,
            "n_b": n2,
        }
    return result


def fdr_correct(
    pairwise_results: Mapping[str, Any],
    qoi: str,
    p_key: str = "ks_p",
) -> Dict[str, Any]:
    """Apply Benjamini-Hochberg FDR correction within a QoI."""
    pair_keys = sorted(pairwise_results.keys())
    raw_p = [pairwise_results[pk][qoi][p_key] for pk in pair_keys]
    _, corrected_p, _, _ = multipletests(raw_p, method="fdr_bh")

    out = {}
    for i, pk in enumerate(pair_keys):
        entry = dict(pairwise_results[pk][qoi])
        entry[f"{p_key}_fdr"] = float(corrected_p[i])
        out[pk] = {qoi: entry}
    return out


def enhancement_trend(
    eq_scenarios: Sequence[Mapping[str, Any]],
    qois: Sequence[str],
    n_boot_qr: int = 200,
) -> Dict[str, Any]:
    """Block 3: Kruskal-Wallis, JT, and quantile regression for enhancement sweep."""
    rng = np.random.default_rng(0)
    result = {}

    enhancements = [s["enhancement"] for s in eq_scenarios]
    groups_by_qoi = {
        qoi: [s["qois"][qoi][np.isfinite(s["qois"][qoi])] for s in eq_scenarios]
        for qoi in qois
    }

    for qoi in qois:
        groups = groups_by_qoi[qoi]
        entry: Dict[str, Any] = {}

        # Kruskal-Wallis
        kw_H, kw_p = stats.kruskal(*groups)
        entry["kw_H"] = float(kw_H)
        entry["kw_p"] = float(kw_p)

        # Jonckheere-Terpstra (only for JT-eligible QoIs)
        if qoi in JT_QOIS:
            if qoi == "lid_fraction":
                alt = "increasing"
            elif qoi == "D_cond_km":
                alt = "two-sided"
            else:
                alt = "decreasing"
            jt_J, jt_p = _jonckheere_terpstra(groups, alternative=alt)
            entry["jt_J"] = jt_J
            entry["jt_p"] = jt_p
            entry["jt_alternative"] = alt

        # Quantile regression — descriptive slopes with bootstrap CIs
        y_all = np.concatenate(groups)
        x_all = np.concatenate([
            np.full(len(g), enh) for g, enh in zip(groups, enhancements)
        ])
        X = np.column_stack([np.ones_like(x_all), x_all])

        for tau_label, tau in [("P5", 0.05), ("P50", 0.50), ("P95", 0.95)]:
            try:
                model = QuantReg(y_all, X)
                fit = model.fit(q=tau, max_iter=1000)
                slope = float(fit.params[1])
            except Exception:
                slope = np.nan

            # Bootstrap CI on slope (suppress QR convergence warnings)
            boot_slopes = []
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for _ in range(n_boot_qr):
                    idx = [rng.choice(len(g), len(g), replace=True) for g in groups]
                    y_b = np.concatenate([g[i] for g, i in zip(groups, idx)])
                    x_b = np.concatenate([
                        np.full(len(i), enh) for i, enh in zip(idx, enhancements)
                    ])
                    X_b = np.column_stack([np.ones_like(x_b), x_b])
                    try:
                        fit_b = QuantReg(y_b, X_b).fit(q=tau, max_iter=500)
                        boot_slopes.append(float(fit_b.params[1]))
                    except Exception:
                        pass

            if boot_slopes:
                ci_low, ci_high = np.percentile(boot_slopes, [2.5, 97.5])
            else:
                ci_low, ci_high = np.nan, np.nan

            entry[f"qr_slope_{tau_label}"] = slope
            entry[f"qr_slope_{tau_label}_ci_low"] = float(ci_low)
            entry[f"qr_slope_{tau_label}_ci_high"] = float(ci_high)

        result[qoi] = entry
    return result


def parameter_ranking(
    data: Mapping[str, Any],
    qois: Sequence[str],
) -> Dict[str, Any]:
    """Block 4: Spearman rank correlations for each param -> QoI."""
    n_params = len(data["params"])

    result = {}
    for qoi in qois:
        y = data["qois"][qoi]
        qoi_result = {}
        for param_name, x in data["params"].items():
            mask = np.isfinite(x) & np.isfinite(y)
            if np.sum(mask) < 10:
                continue
            rho, p = stats.spearmanr(x[mask], y[mask])
            qoi_result[param_name] = {
                "rho": float(rho),
                "p": float(p),
                "significant": p < (0.001 / n_params),
            }
        # Assign ranks by |rho|
        sorted_params = sorted(
            qoi_result.keys(), key=lambda k: abs(qoi_result[k]["rho"]), reverse=True
        )
        for rank, param_name in enumerate(sorted_params, start=1):
            qoi_result[param_name]["rank"] = rank
        result[qoi] = qoi_result
    return result


def ranking_concordance(
    rankings_by_scenario: Mapping[str, Any],
    qois: Sequence[str],
) -> Dict[str, Any]:
    """Block 4: Kendall's W across scenarios for each QoI."""
    scenario_names = list(rankings_by_scenario.keys())
    result = {}
    for qoi in qois:
        # Collect parameter names present in all scenarios
        all_param_sets = [
            set(rankings_by_scenario[s][qoi].keys()) for s in scenario_names
        ]
        common_params = sorted(set.intersection(*all_param_sets))
        if len(common_params) < 2:
            continue
        # Build rankings matrix: (k_raters, n_items)
        matrix = np.array([
            [rankings_by_scenario[s][qoi][p]["rank"] for p in common_params]
            for s in scenario_names
        ], dtype=float)
        W, chi2, p = _kendall_w(matrix)
        result[qoi] = {
            "W": W,
            "chi2": chi2,
            "p": p,
            "n_params": len(common_params),
            "n_scenarios": len(scenario_names),
        }
    return result


def bootstrap_convergence(
    data: Mapping[str, Any],
    qois: Sequence[str],
    subsample_sizes: Optional[Sequence[int]] = None,
    n_boot: int = 1000,
) -> Dict[str, Any]:
    """Block 5: bootstrap CI width vs sample size."""
    if subsample_sizes is None:
        subsample_sizes = CONVERGENCE_SCHEDULE
    rng = np.random.default_rng(0)
    result: Dict[str, Any] = {}

    for qoi in qois:
        arr = data["qois"][qoi]
        arr = arr[np.isfinite(arr)]
        qoi_result = {}
        for n_sub in subsample_sizes:
            if n_sub > len(arr):
                continue
            sub = rng.choice(arr, n_sub, replace=False)
            ci_median = _bca_ci(sub, np.median, n_boot, rng=rng)
            ci_p5 = _bca_ci(sub, lambda x: np.percentile(x, 5), n_boot, rng=rng)
            ci_p95 = _bca_ci(sub, lambda x: np.percentile(x, 95), n_boot, rng=rng)
            qoi_result[n_sub] = {
                "median": float(np.median(sub)),
                "ci_median_low": ci_median[0],
                "ci_median_high": ci_median[1],
                "ci_median_width": ci_median[1] - ci_median[0],
                "P5": float(np.percentile(sub, 5)),
                "ci_P5_low": ci_p5[0],
                "ci_P5_high": ci_p5[1],
                "P95": float(np.percentile(sub, 95)),
                "ci_P95_low": ci_p95[0],
                "ci_P95_high": ci_p95[1],
            }
        # Determine sample size where CI width < 1 km
        threshold_n = None
        for n_sub in sorted(qoi_result.keys()):
            if qoi_result[n_sub]["ci_median_width"] < 1.0:
                threshold_n = n_sub
                break
        qoi_result["convergence_threshold_n"] = threshold_n
        result[qoi] = qoi_result

    # Conductive fraction convergence
    if "lid_fraction" in data["qois"]:
        lf = data["qois"]["lid_fraction"]
        cond = (lf >= LID_FRACTION_COND_THRESHOLD).astype(float)
        cf_result = {}
        for n_sub in subsample_sizes:
            if n_sub > len(cond):
                continue
            sub = rng.choice(cond, n_sub, replace=False)
            ci = _bca_ci(sub, np.mean, n_boot, rng=rng)
            cf_result[n_sub] = {
                "fraction": float(np.mean(sub)),
                "ci_low": ci[0],
                "ci_high": ci[1],
                "ci_width": ci[1] - ci[0],
            }
        result["conductive_fraction"] = cf_result

    return result


def shell_structure(
    data_by_scenario: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    """Block 6: shell structure partitioning with subpopulation analysis."""
    scenario_names = list(data_by_scenario.keys())
    result: Dict[str, Any] = {
        "whole_pop": {},
        "convective_subpop": {},
        "conductive_fractions": {},
        "dcond_dconv_correlation": {},
    }

    # Conductive fractions with bootstrap CIs
    rng = np.random.default_rng(0)
    for name, data in data_by_scenario.items():
        lf = data["qois"]["lid_fraction"]
        frac = float(np.mean(lf >= LID_FRACTION_COND_THRESHOLD))
        ci = _bca_ci(
            (lf >= LID_FRACTION_COND_THRESHOLD).astype(float),
            np.mean, 2000, rng=rng,
        )
        result["conductive_fractions"][name] = {
            "fraction": frac, "ci_low": ci[0], "ci_high": ci[1],
        }

    # Whole-population pairwise (lid_fraction only — Block 2 handles thickness/D_cond)
    for i, name_a in enumerate(scenario_names):
        for name_b in scenario_names[i + 1:]:
            pair_key = f"{name_a}_vs_{name_b}"
            result["whole_pop"][pair_key] = pairwise_comparison(
                data_by_scenario[name_a],
                data_by_scenario[name_b],
                ["lid_fraction"],
            )

    # Convective subpopulation
    conv_data = {}
    for name, data in data_by_scenario.items():
        mask = data["qois"]["lid_fraction"] < LID_FRACTION_COND_THRESHOLD
        conv_data[name] = {
            "qois": {
                qoi: data["qois"][qoi][mask]
                for qoi in ALL_QOIS
                if qoi in data["qois"]
            },
        }

    conv_qois = ["thickness_km", "D_cond_km", "D_conv_km", "Ra", "Nu"]
    for i, name_a in enumerate(scenario_names):
        for name_b in scenario_names[i + 1:]:
            pair_key = f"{name_a}_vs_{name_b}"
            result["convective_subpop"][pair_key] = pairwise_comparison(
                conv_data[name_a], conv_data[name_b], conv_qois,
            )

    # FDR correction on convective subpop pairwise tests (same scope as Block 2)
    subpop_pairs = {k: v for k, v in result["convective_subpop"].items() if "_vs_" in k}
    for qoi in conv_qois:
        for p_key in ("ks_p", "mw_p"):
            corrected = fdr_correct(subpop_pairs, qoi, p_key)
            for pk, entry in corrected.items():
                result["convective_subpop"][pk][qoi].update(entry[qoi])

    # Ra/Nu conditional stats per scenario
    for name, cd in conv_data.items():
        for qoi in ("Ra", "Nu"):
            arr = cd["qois"].get(qoi, np.array([]))
            finite = arr[np.isfinite(arr)]
            if len(finite) == 0:
                continue
            log_arr = np.log10(finite[finite > 0])
            result["convective_subpop"].setdefault(f"{name}_stats", {})[qoi] = {
                "mean": float(np.mean(finite)),
                "median": float(np.median(finite)),
                "IQR": float(np.percentile(finite, 75) - np.percentile(finite, 25)),
                "log10_mean": float(np.mean(log_arr)) if len(log_arr) > 0 else None,
                "log10_median": float(np.median(log_arr)) if len(log_arr) > 0 else None,
                "log10_IQR": float(np.percentile(log_arr, 75) - np.percentile(log_arr, 25)) if len(log_arr) > 0 else None,
            }

    # D_cond vs D_conv Pearson r per scenario
    for name, cd in conv_data.items():
        dc = cd["qois"].get("D_cond_km", np.array([]))
        dv = cd["qois"].get("D_conv_km", np.array([]))
        mask = np.isfinite(dc) & np.isfinite(dv)
        if np.sum(mask) > 10:
            r, p = stats.pearsonr(dc[mask], dv[mask])
            result["dcond_dconv_correlation"][name] = {
                "pearson_r": float(r), "p": float(p),
            }

    return result


def run_all(
    scenario_paths: Mapping[str, str],
    output_dir: str = "results/thesis_stats",
) -> Dict[str, Any]:
    """Orchestrate all 6 analysis blocks and return combined results."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Load all scenarios
    scenarios = {name: load_scenario(path) for name, path in scenario_paths.items()}
    scenario_names = list(scenarios.keys())

    results: Dict[str, Any] = {"scenarios": {}}

    # Block 1: Descriptive summary
    for name, data in scenarios.items():
        results["scenarios"][name] = {
            "descriptive": descriptive_summary(data),
            "metadata": data["metadata"],
        }

    # Block 2: Pairwise comparison (whole-pop QoIs only)
    results["pairwise"] = {}
    for i, name_a in enumerate(scenario_names):
        for name_b in scenario_names[i + 1:]:
            pair_key = f"{name_a}_vs_{name_b}"
            pw = pairwise_comparison(
                scenarios[name_a], scenarios[name_b], list(WHOLE_POP_QOIS),
            )
            results["pairwise"][pair_key] = pw

    # FDR correction within each QoI
    results["pairwise_fdr"] = {}
    for qoi in WHOLE_POP_QOIS:
        for p_key in ("ks_p", "mw_p"):
            corrected = fdr_correct(results["pairwise"], qoi, p_key)
            for pk, entry in corrected.items():
                results["pairwise"].setdefault(pk, {}).setdefault(qoi, {}).update(
                    entry[qoi]
                )

    # Block 3: Enhancement trend (equatorial only)
    eq_names = [n for n in scenario_names if "Eq" in n]
    eq_enhancements = {
        "Eq Baseline": 1.0, "Eq Moderate": 1.2, "Eq Strong": 1.5,
    }
    eq_data = [
        {**scenarios[n], "enhancement": eq_enhancements[n]}
        for n in eq_names if n in eq_enhancements
    ]
    if len(eq_data) >= 2:
        results["enhancement_trend"] = enhancement_trend(eq_data, list(JT_QOIS))

    # Block 4: Parameter ranking
    results["parameter_ranking"] = {}
    for name, data in scenarios.items():
        results["parameter_ranking"][name] = parameter_ranking(data, list(WHOLE_POP_QOIS))
    results["ranking_concordance"] = ranking_concordance(
        results["parameter_ranking"], list(WHOLE_POP_QOIS),
    )

    # Block 5: Bootstrap convergence
    results["bootstrap_convergence"] = {}
    for name, data in scenarios.items():
        results["bootstrap_convergence"][name] = bootstrap_convergence(
            data, ["thickness_km"],
        )

    # Block 6: Shell structure
    results["shell_structure"] = shell_structure(scenarios)

    return results


def _to_serializable(obj):
    """Convert numpy types to JSON-serializable Python types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(i) for i in obj]
    return obj


def save_results(results: Dict[str, Any], output_dir: str) -> None:
    """Write JSON + summary CSV."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # JSON
    with (out_path / "comparison_results.json").open("w", encoding="utf-8") as f:
        json.dump(_to_serializable(results), f, indent=2)

    # Summary tables CSV (long format)
    rows = []
    for scenario_name, scenario_data in results.get("scenarios", {}).items():
        desc = scenario_data.get("descriptive", {})
        for qoi, stats_dict in desc.items():
            if not isinstance(stats_dict, dict):
                # scalar like conductive_fraction
                rows.append({
                    "scenario": scenario_name, "qoi": qoi,
                    "statistic": "value", "value": stats_dict,
                    "ci_low": "", "ci_high": "",
                })
                continue
            for stat_name, value in stats_dict.items():
                if stat_name.startswith("ci_") or stat_name == "n":
                    continue
                ci_low = stats_dict.get(f"ci_{stat_name}_low", "")
                ci_high = stats_dict.get(f"ci_{stat_name}_high", "")
                rows.append({
                    "scenario": scenario_name, "qoi": qoi,
                    "statistic": stat_name, "value": value,
                    "ci_low": ci_low, "ci_high": ci_high,
                })

    if rows:
        fieldnames = ["scenario", "qoi", "statistic", "value", "ci_low", "ci_high"]
        with (out_path / "summary_tables.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    # Parameter rankings CSV
    rank_rows = []
    for scenario_name, qoi_ranks in results.get("parameter_ranking", {}).items():
        for qoi, param_dict in qoi_ranks.items():
            for param_name, param_stats in param_dict.items():
                rank_rows.append({
                    "scenario": scenario_name, "qoi": qoi,
                    "parameter": param_name, "rho": param_stats["rho"],
                    "rank": param_stats["rank"],
                    "significant": param_stats["significant"],
                })
    if rank_rows:
        fieldnames = ["scenario", "qoi", "parameter", "rho", "rank", "significant"]
        with (out_path / "parameter_rankings.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rank_rows)


if __name__ == "__main__":
    import os

    logging.basicConfig(level=logging.INFO)
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    paths = {
        label: os.path.join(results_dir, filename)
        for label, filename in DEFAULT_SCENARIOS.items()
    }
    results = run_all(paths)
    save_results(results, os.path.join(results_dir, "thesis_stats"))
    print("Done. Results saved to results/thesis_stats/")
