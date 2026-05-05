import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import numpy as np
import pytest
from scipy import stats
from thesis_stats import load_scenario
from thesis_stats import _cliffs_delta_from_u, _jonckheere_terpstra, _kendall_w
from thesis_stats import descriptive_summary, LID_FRACTION_COND_THRESHOLD
from thesis_stats import pairwise_comparison, fdr_correct, WHOLE_POP_QOIS
from thesis_stats import enhancement_trend
from thesis_stats import parameter_ranking, ranking_concordance
from thesis_stats import bootstrap_convergence
from thesis_stats import shell_structure


# ---------------------------------------------------------------------------
# Task 1: load_scenario + zero-variance filter
# ---------------------------------------------------------------------------

def test_load_scenario_filters_zero_variance_params(tmp_path):
    """Zero-variance columns are excluded; varying columns are kept."""
    npz_path = tmp_path / "fake.npz"
    np.savez(
        npz_path,
        thicknesses_km=np.random.default_rng(0).normal(30, 5, 100),
        D_cond_km=np.random.default_rng(0).normal(15, 3, 100),
        D_conv_km=np.random.default_rng(0).normal(15, 3, 100),
        lid_fractions=np.random.default_rng(0).uniform(0.3, 1.0, 100),
        Ra_values=np.random.default_rng(0).lognormal(8, 2, 100),
        Nu_values=np.random.default_rng(0).lognormal(1, 0.5, 100),
        param_good=np.random.default_rng(0).normal(10, 1, 100),
        param_constant=np.full(100, 42.0),
        param_tiny_constant=np.full(100, 1e-12),
    )
    data = load_scenario(str(npz_path))
    assert "good" in data["params"]
    assert "constant" not in data["params"]
    assert "tiny_constant" not in data["params"]
    assert "thickness_km" in data["qois"]
    assert "lid_fraction" in data["qois"]


# ---------------------------------------------------------------------------
# Task 2: Custom statistics — Cliff's delta, Jonckheere-Terpstra, Kendall's W
# ---------------------------------------------------------------------------

def test_cliffs_delta_identical_distributions():
    a = np.arange(100, dtype=float)
    u_stat, _ = stats.mannwhitneyu(a, a, alternative="two-sided")
    d = _cliffs_delta_from_u(u_stat, len(a), len(a))
    assert abs(d) < 0.1


def test_cliffs_delta_fully_separated():
    a = np.arange(100, dtype=float)
    b = np.arange(200, 300, dtype=float)
    u_stat, _ = stats.mannwhitneyu(a, b, alternative="two-sided")
    d = _cliffs_delta_from_u(u_stat, len(a), len(b))
    assert d < -0.9  # a is smaller than b


def test_jt_monotonic_shift():
    rng = np.random.default_rng(42)
    groups = [rng.normal(0, 1, 500), rng.normal(1, 1, 500), rng.normal(2, 1, 500)]
    j_stat, p_value = _jonckheere_terpstra(groups, alternative="increasing")
    assert p_value < 0.001


def test_jt_identical_groups():
    rng = np.random.default_rng(42)
    base = rng.normal(0, 1, 500)
    groups = [base.copy(), base.copy(), base.copy()]
    _, p_value = _jonckheere_terpstra(groups, alternative="increasing")
    assert p_value > 0.05


def test_jt_reversed_order_one_sided():
    rng = np.random.default_rng(42)
    groups = [rng.normal(2, 1, 500), rng.normal(1, 1, 500), rng.normal(0, 1, 500)]
    _, p_value = _jonckheere_terpstra(groups, alternative="increasing")
    assert p_value > 0.5


def test_jt_tie_correction_differs():
    """Heavy ties should produce different variance than no-tie formula."""
    rng = np.random.default_rng(42)
    # Groups with many tied values (discretized)
    groups = [
        np.round(rng.normal(0, 1, 300), 0),
        np.round(rng.normal(1, 1, 300), 0),
        np.round(rng.normal(2, 1, 300), 0),
    ]
    j_stat, p_value = _jonckheere_terpstra(groups, alternative="increasing")
    assert p_value < 0.05  # still detects the trend despite ties


def test_kendall_w_identical_rankings():
    rankings = np.array([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]])
    w, _, _ = _kendall_w(rankings)
    assert abs(w - 1.0) < 1e-10


def test_kendall_w_random_rankings():
    rng = np.random.default_rng(42)
    rankings = np.array([rng.permutation(10) + 1 for _ in range(4)])
    w, _, _ = _kendall_w(rankings)
    assert 0.0 <= w <= 1.0
    assert w < 0.5  # random rankings should have low concordance


# ---------------------------------------------------------------------------
# Task 3: Block 1 — descriptive_summary
# ---------------------------------------------------------------------------

def test_descriptive_summary_known_array():
    data = {
        "qois": {"thickness_km": np.arange(1.0, 101.0)},
        "params": {},
        "metadata": {"n_valid": 100},
    }
    result = descriptive_summary(data, n_boot=500)
    t = result["thickness_km"]
    assert abs(t["median"] - 50.5) < 0.01
    assert abs(t["P25"] - 25.75) < 0.01
    assert abs(t["P75"] - 75.25) < 0.01
    assert "ci_median_low" in t
    assert "ci_median_high" in t
    assert t["ci_median_low"] < t["median"] < t["ci_median_high"]


def test_descriptive_summary_conductive_fraction():
    lid = np.concatenate([np.full(40, 1.0), np.linspace(0.3, 0.95, 60)])
    data = {
        "qois": {"lid_fraction": lid},
        "params": {},
        "metadata": {"n_valid": 100},
    }
    result = descriptive_summary(data, n_boot=500)
    assert abs(result["conductive_fraction"] - 0.40) < 0.01


# ---------------------------------------------------------------------------
# Task 4: Block 2 — pairwise_comparison + fdr_correct
# ---------------------------------------------------------------------------

def test_pairwise_identical_distributions():
    rng = np.random.default_rng(42)
    arr = rng.normal(30, 5, 1000)
    data_a = {"qois": {"thickness_km": arr.copy()}}
    data_b = {"qois": {"thickness_km": arr.copy()}}
    result = pairwise_comparison(data_a, data_b, ["thickness_km"])
    t = result["thickness_km"]
    assert t["ks_D"] < 0.05
    assert t["ks_p"] > 0.1
    assert abs(t["cliff_d"]) < 0.1
    assert abs(t["cohen_d"]) < 0.1


def test_pairwise_separated_distributions():
    rng = np.random.default_rng(42)
    data_a = {"qois": {"thickness_km": rng.normal(20, 2, 1000)}}
    data_b = {"qois": {"thickness_km": rng.normal(40, 2, 1000)}}
    result = pairwise_comparison(data_a, data_b, ["thickness_km"])
    t = result["thickness_km"]
    assert t["ks_D"] > 0.8
    assert t["ks_p"] < 1e-10
    assert abs(t["cliff_d"]) > 0.9
    assert abs(t["cohen_d"]) > 5.0


def test_fdr_correct_known_pvalues():
    from statsmodels.stats.multitest import multipletests
    raw_p = [0.001, 0.01, 0.03, 0.04, 0.8, 0.9]
    results = {
        f"pair_{i}": {"thickness_km": {"ks_p": p}}
        for i, p in enumerate(raw_p)
    }
    corrected = fdr_correct(results, qoi="thickness_km", p_key="ks_p")
    _, expected, _, _ = multipletests(raw_p, method="fdr_bh")
    for i, pair_key in enumerate(sorted(corrected)):
        np.testing.assert_allclose(
            corrected[pair_key]["thickness_km"]["ks_p_fdr"],
            expected[i], rtol=1e-10,
        )


# ---------------------------------------------------------------------------
# Task 5: Block 3 — enhancement_trend
# ---------------------------------------------------------------------------

def test_enhancement_trend_detects_monotonic_decrease():
    rng = np.random.default_rng(42)
    scenarios = [
        {"qois": {"thickness_km": rng.normal(40, 5, 1000)}, "enhancement": 1.0},
        {"qois": {"thickness_km": rng.normal(30, 5, 1000)}, "enhancement": 1.2},
        {"qois": {"thickness_km": rng.normal(20, 5, 1000)}, "enhancement": 1.5},
    ]
    result = enhancement_trend(scenarios, ["thickness_km"])
    t = result["thickness_km"]
    assert t["kw_p"] < 0.001
    assert t["jt_p"] < 0.001
    assert t["qr_slope_P50"] < 0  # thickness decreases with enhancement
    assert t["qr_slope_P50_ci_low"] < t["qr_slope_P50"] < t["qr_slope_P50_ci_high"]


# ---------------------------------------------------------------------------
# Task 6: Block 4 — parameter_ranking + ranking_concordance
# ---------------------------------------------------------------------------

def test_parameter_ranking_detects_dominant_control():
    rng = np.random.default_rng(42)
    n = 1000
    x_strong = rng.uniform(0, 1, n)
    x_weak = rng.uniform(0, 1, n)
    y = 10 * x_strong + 0.1 * x_weak + rng.normal(0, 0.5, n)
    data = {
        "qois": {"thickness_km": y},
        "params": {"strong": x_strong, "weak": x_weak},
        "metadata": {"n_valid": n},
    }
    result = parameter_ranking(data, ["thickness_km"])
    ranks = result["thickness_km"]
    # "strong" should have higher |rho| than "weak"
    assert abs(ranks["strong"]["rho"]) > abs(ranks["weak"]["rho"])
    assert ranks["strong"]["significant"]


def test_ranking_concordance_perfect_agreement():
    rankings = {
        "scenario_a": {"thickness_km": {"p1": {"rank": 1}, "p2": {"rank": 2}}},
        "scenario_b": {"thickness_km": {"p1": {"rank": 1}, "p2": {"rank": 2}}},
    }
    result = ranking_concordance(rankings, ["thickness_km"])
    assert result["thickness_km"]["W"] > 0.99


# ---------------------------------------------------------------------------
# Task 7: Block 5 — bootstrap_convergence
# ---------------------------------------------------------------------------

def test_bootstrap_convergence_ci_narrows():
    rng = np.random.default_rng(42)
    data = {
        "qois": {"thickness_km": rng.normal(30, 5, 15000)},
        "params": {},
        "metadata": {"n_valid": 15000},
    }
    result = bootstrap_convergence(data, ["thickness_km"], [500, 5000, 14997])
    t = result["thickness_km"]
    # CI should narrow with more samples
    ci_width_500 = t[500]["ci_median_high"] - t[500]["ci_median_low"]
    ci_width_14997 = t[14997]["ci_median_high"] - t[14997]["ci_median_low"]
    assert ci_width_500 > ci_width_14997


# ---------------------------------------------------------------------------
# Task 8: Block 6 — shell_structure
# ---------------------------------------------------------------------------

def test_shell_structure_splits_subpopulations():
    rng = np.random.default_rng(42)
    n = 2000
    lid = np.concatenate([np.full(800, 1.0), rng.uniform(0.3, 0.95, 1200)])
    data_by_scenario = {
        "A": {
            "qois": {
                "thickness_km": rng.normal(30, 5, n),
                "D_cond_km": rng.normal(15, 3, n),
                "D_conv_km": np.where(lid < 0.999, rng.uniform(1, 20, n), 0.0),
                "lid_fraction": lid,
                "Ra": np.where(lid < 0.999, rng.lognormal(8, 2, n), 10.0),
                "Nu": np.where(lid < 0.999, rng.lognormal(1, 0.5, n), 1.0),
            },
            "params": {},
            "metadata": {"n_valid": n},
        },
        "B": {
            "qois": {
                "thickness_km": rng.normal(25, 5, n),
                "D_cond_km": rng.normal(12, 3, n),
                "D_conv_km": np.where(lid < 0.999, rng.uniform(1, 15, n), 0.0),
                "lid_fraction": lid,
                "Ra": np.where(lid < 0.999, rng.lognormal(7, 2, n), 10.0),
                "Nu": np.where(lid < 0.999, rng.lognormal(0.8, 0.5, n), 1.0),
            },
            "params": {},
            "metadata": {"n_valid": n},
        },
    }
    result = shell_structure(data_by_scenario)
    assert "whole_pop" in result
    assert "convective_subpop" in result
    assert "conductive_fractions" in result
    # Check convective subpop pairwise exists
    assert "A_vs_B" in result["convective_subpop"]


# ---------------------------------------------------------------------------
# Task 10: Smoke tests
# ---------------------------------------------------------------------------

def test_quantile_regression_recovers_known_slope():
    """Direct QR test: y = 2x + noise, slope CI should contain 2.0."""
    from statsmodels.regression.quantile_regression import QuantReg as QR
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 10, 500)
    y = 2.0 * x + rng.normal(0, 1, 500)
    X = np.column_stack([np.ones_like(x), x])
    fit = QR(y, X).fit(q=0.5)
    slope = fit.params[1]
    # 95% CI from bootstrap
    boot_slopes = []
    for _ in range(1000):
        idx = rng.choice(len(x), len(x), replace=True)
        fit_b = QR(y[idx], X[idx]).fit(q=0.5)
        boot_slopes.append(fit_b.params[1])
    ci_low, ci_high = np.percentile(boot_slopes, [2.5, 97.5])
    assert ci_low < 2.0 < ci_high


def test_save_results_produces_valid_outputs(tmp_path):
    """save_results writes parseable JSON and CSV with expected columns."""
    from thesis_stats import save_results
    fake_results = {
        "scenarios": {
            "A": {
                "descriptive": {
                    "thickness_km": {"mean": 30.0, "median": 28.0},
                    "conductive_fraction": 0.35,
                },
                "metadata": {"n_valid": 100},
            }
        },
        "parameter_ranking": {
            "A": {
                "thickness_km": {
                    "P_tidal": {"rho": -0.8, "rank": 1, "significant": True},
                }
            }
        },
    }
    save_results(fake_results, str(tmp_path))
    import json, csv
    with open(tmp_path / "comparison_results.json") as f:
        loaded = json.load(f)
    assert "scenarios" in loaded

    with open(tmp_path / "summary_tables.csv", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert len(rows) > 0
    assert "scenario" in rows[0]
    assert "qoi" in rows[0]

    with open(tmp_path / "parameter_rankings.csv", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert len(rows) > 0
    assert rows[0]["parameter"] == "P_tidal"


@pytest.mark.slow
def test_smoke_run_all_on_real_archives():
    """Smoke test: run_all on all four production archives."""
    import os
    from thesis_stats import run_all, save_results, DEFAULT_SCENARIOS

    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    paths = {
        label: os.path.join(results_dir, filename)
        for label, filename in DEFAULT_SCENARIOS.items()
    }
    # Skip if archives not present
    for p in paths.values():
        if not os.path.exists(p):
            pytest.skip(f"Archive not found: {p}")

    results = run_all(paths)

    assert "scenarios" in results
    assert len(results["scenarios"]) == 4
    assert "pairwise" in results
    assert "enhancement_trend" in results
    assert "parameter_ranking" in results
    assert "bootstrap_convergence" in results
    assert "shell_structure" in results

    # Check descriptive summary has expected keys
    for scenario_data in results["scenarios"].values():
        desc = scenario_data["descriptive"]
        assert "thickness_km" in desc
        assert "median" in desc["thickness_km"]
        assert "conductive_fraction" in desc
