import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import csv
import numpy as np
import pytest

import sobol_workflow
from run_sobol_suite import write_csv
from sobol_workflow import (
    build_salib_problem,
    compute_sobol_indices,
    default_convergence_schedule,
    evaluate_sobol_design,
    expected_sobol_rows,
    generate_sobol_design,
    get_primary_prior_specs,
    is_power_of_two,
    map_unit_sample_to_model,
    sobol_results_to_rows,
    _truncated_normal_ppf,
)


def test_build_salib_problem_grouping_changes_effective_dimension():
    problem_params = build_salib_problem("global_audited", grouped=False)
    problem_groups = build_salib_problem("global_audited", grouped=True)

    assert problem_params["num_vars"] == 10
    assert "groups" not in problem_params
    assert len(problem_groups["groups"]) == problem_groups["num_vars"]
    assert len(dict.fromkeys(problem_groups["groups"])) == 5


def test_map_unit_sample_global_audited_stays_inside_audited_bounds():
    specs = get_primary_prior_specs("global_audited")
    mapped = map_unit_sample_to_model([0.5] * len(specs), "global_audited")

    for spec in specs:
        value = mapped.prior_inputs[spec.name]
        assert spec.low <= value <= spec.high

    solver_params = mapped.solver_params
    assert 5.0e-5 <= solver_params["d_grain"] <= 3.0e-3
    assert 80.0e3 <= solver_params["D_H2O"] <= 200.0e3
    assert 2.0e9 <= solver_params["mu_ice"] <= 5.0e9
    assert solver_params["f_salt"] == 0.0
    assert solver_params["B_k"] == 1.0
    assert solver_params["T_phi"] == 150.0
    assert mapped.diagnostics["q_basal_effective_mW_m2"] >= 0.0


def test_equatorial_enhancement_scales_only_tidal_component():
    unit_sample = [0.35] * len(get_primary_prior_specs("equatorial_baseline"))
    baseline = map_unit_sample_to_model(unit_sample, "equatorial_baseline")
    strong = map_unit_sample_to_model(unit_sample, "equatorial_strong")

    assert "eq_enhancement" not in baseline.solver_params
    assert strong.diagnostics["enhancement_factor"] == 1.5
    assert baseline.solver_params["T_surf"] == strong.solver_params["T_surf"]
    assert baseline.solver_params["epsilon_0"] == strong.solver_params["epsilon_0"]
    assert baseline.diagnostics["q_radiogenic_mW_m2"] == strong.diagnostics["q_radiogenic_mW_m2"]
    np.testing.assert_allclose(
        strong.diagnostics["q_silicate_tidal_mW_m2"],
        1.5 * baseline.diagnostics["q_silicate_tidal_mW_m2"],
        rtol=1.0e-12,
    )


def test_expected_sobol_rows_respects_grouped_dimension():
    problem_params = build_salib_problem("global_audited", grouped=False)
    problem_groups = build_salib_problem("global_audited", grouped=True)

    assert expected_sobol_rows(problem_params, 512, calc_second_order=False) == 512 * 12
    assert expected_sobol_rows(problem_groups, 512, calc_second_order=False) == 512 * 7


def test_default_convergence_schedule_tracks_powers_of_two():
    schedule = default_convergence_schedule(512)
    assert schedule == [128, 256, 512]
    assert all(is_power_of_two(value) for value in schedule)


def test_truncated_normal_ppf_stays_inside_bounds_at_extreme_quantiles():
    low = 80.0
    high = 120.0
    for u in (0.001, 0.01, 0.5, 0.99, 0.999):
        value = _truncated_normal_ppf(u, mean=104.0, sigma=7.0, low=low, high=high)
        assert low <= value <= high


def test_end_to_end_sobol_pipeline_runs_with_mocked_evaluator(monkeypatch):
    pytest.importorskip("SALib")

    def fake_evaluate_fixed_params(solver_params, config, physical_output_policy):
        x = solver_params["T_surf"] / 100.0
        y = solver_params["epsilon_0"] * 1.0e6
        thickness = 10.0 + 2.0 * x + 3.0 * y
        return {
            "numerical_success": 1.0,
            "physical_flag": 1.0,
            "valid_flag": 1.0,
            "subcritical_flag": 0.0,
            "convective_flag": 1.0 if thickness > 15.0 else 0.0,
            "thickness_km": thickness,
            "D_cond_km": 0.5 * thickness,
            "D_conv_km": 0.5 * thickness,
            "lid_fraction": 0.5,
            "Ra": 1.0e5 + thickness,
            "Nu": 1.0 + 0.1 * thickness,
            "q_radiogenic_mW_m2": 5.0,
            "q_silicate_tidal_mW_m2": 10.0,
            "q_basal_effective_mW_m2": 15.0,
        }

    monkeypatch.setattr(sobol_workflow, "_evaluate_fixed_params", fake_evaluate_fixed_params)

    problem = build_salib_problem("global_audited", grouped=False)
    unit_design = generate_sobol_design(problem, 8, seed=42)
    evaluation = evaluate_sobol_design(
        unit_design,
        "global_audited",
        config=object(),
        n_workers=1,
        physical_output_policy="keep",
        verbose=False,
    )
    results = compute_sobol_indices(
        problem,
        evaluation["outputs"],
        output_names=["thickness_km"],
        base_sample_sizes=[8],
        num_resamples=16,
        seed=42,
    )

    assert "final" in results["thickness_km"]
    assert len(results["thickness_km"]["final"]["Si"]["ST"]) == problem["num_vars"]


def test_csv_export_preserves_sobol_rows(tmp_path):
    problem = {
        "num_vars": 2,
        "names": ["a", "b"],
        "bounds": [[0.0, 1.0], [0.0, 1.0]],
    }
    sobol_results = {
        "thickness_km": {
            "convergence": [
                {
                    "N": 8,
                    "Si": {
                        "S1": np.array([0.1, 0.2]),
                        "S1_conf": np.array([0.01, 0.02]),
                        "ST": np.array([0.3, 0.4]),
                        "ST_conf": np.array([0.03, 0.04]),
                    },
                }
            ]
        }
    }

    rows = sobol_results_to_rows(problem, sobol_results)
    csv_path = tmp_path / "sobol_rows.csv"
    write_csv(rows, csv_path)

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = list(csv.DictReader(handle))

    assert len(reader) == 2
    assert reader[0]["output"] == "thickness_km"
    assert reader[0]["factor"] == "a"
    assert reader[0]["S1"] == "0.1"
    assert reader[1]["factor"] == "b"
    assert reader[1]["ST"] == "0.4"
