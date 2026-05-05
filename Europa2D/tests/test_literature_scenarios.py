import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import src

import pytest

from literature_scenarios import DEFAULT_SCENARIO, get_scenario, list_scenarios


def test_default_scenario_is_uniform_transport():
    """Neutral baseline must be uniform_transport (global Howell 2021 params)."""
    from literature_scenarios import DEFAULT_SCENARIO
    assert DEFAULT_SCENARIO == "uniform_transport", (
        f"DEFAULT_SCENARIO={DEFAULT_SCENARIO!r}, expected 'uniform_transport'"
    )


def test_list_scenarios_contains_core_literature_cases():
    scenarios = set(list_scenarios())
    assert "uniform_transport" in scenarios
    assert "soderlund2014_equator" in scenarios
    assert "lemasquerier2023_polar" in scenarios
    assert "lemasquerier2023_polar_strong" in scenarios


def test_soderlund_preset_has_expected_endpoint_ratio():
    """q_star=0.4, equator_enhanced -> a=0.545 -> q_eq/q_pole = 1.55."""
    scenario = get_scenario("soderlund2014_equator")
    profile = scenario.build_profile(
        T_eq=110.0, epsilon_eq=6e-6, epsilon_pole=1.2e-5, q_ocean_mean=0.02,
    )
    q_eq, q_pole = profile.ocean_endpoint_fluxes()
    assert q_eq / q_pole == pytest.approx(1.545, rel=0.01)


def test_lemasquerier_conservative_endpoint_ratio():
    """q_star=0.455, polar_enhanced -> a=0.536 -> q_pole/q_eq = 1.54."""
    scenario = get_scenario("lemasquerier2023_polar")
    profile = scenario.build_profile(
        T_eq=110.0, epsilon_eq=6e-6, epsilon_pole=1.2e-5, q_ocean_mean=0.02,
    )
    assert profile.ocean_endpoint_ratio() == pytest.approx(1.54, rel=0.01)


def test_strong_lemasquerier_stronger_than_conservative():
    conservative = get_scenario("lemasquerier2023_polar")
    strong = get_scenario("lemasquerier2023_polar_strong")
    c_prof = conservative.build_profile(110.0, 6e-6, 1.2e-5, 0.02)
    s_prof = strong.build_profile(110.0, 6e-6, 1.2e-5, 0.02)
    assert s_prof.ocean_endpoint_ratio() > c_prof.ocean_endpoint_ratio()


def test_scenarios_use_q_star_not_ocean_amplitude():
    """All scenarios should set q_star, not ocean_amplitude."""
    for name in list_scenarios():
        scenario = get_scenario(name)
        assert hasattr(scenario, 'q_star')


def test_surface_presets_exist():
    """Named surface temperature presets for sensitivity analysis."""
    from literature_scenarios import SURFACE_PRESETS

    assert "ashkenazy_low_q" in SURFACE_PRESETS
    assert "ashkenazy_high_q" in SURFACE_PRESETS
    assert "legacy_110_52" in SURFACE_PRESETS

    low = SURFACE_PRESETS["ashkenazy_low_q"]
    assert low.T_eq == 96.0
    assert low.T_floor == 46.0

    high = SURFACE_PRESETS["ashkenazy_high_q"]
    assert high.T_eq == 96.0
    assert high.T_floor == 53.0

    legacy = SURFACE_PRESETS["legacy_110_52"]
    assert legacy.T_eq == 110.0
    assert legacy.T_floor == 52.0
