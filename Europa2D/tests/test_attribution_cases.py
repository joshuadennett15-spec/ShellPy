import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import src

import numpy as np

from attribution_cases import ATTRIBUTION_CASES, build_paired_attribution_profiles
from latitude_profile import LatitudeProfile


def _sample_profile() -> LatitudeProfile:
    return LatitudeProfile(
        T_eq=104.0,
        T_floor=46.0,
        epsilon_eq=6e-6,
        epsilon_pole=1.2e-5,
        q_ocean_mean=0.02,
        ocean_pattern="polar_enhanced",
        q_star=0.455,
        mantle_tidal_fraction=0.5,
        tidal_pattern="mantle_core",
        surface_pattern="latitude",
        strict_q_star=False,
    )


def test_build_paired_attribution_profiles_returns_all_cases():
    cases = build_paired_attribution_profiles(_sample_profile())
    assert tuple(cases.keys()) == ATTRIBUTION_CASES


def test_paired_cases_toggle_only_targeted_forcing_family():
    source = _sample_profile()
    cases = build_paired_attribution_profiles(source)

    baseline = cases["baseline"]
    surface_only = cases["surface_only"]
    ocean_only = cases["ocean_only"]
    strain_only = cases["strain_only"]

    # Surface toggle
    assert baseline.surface_pattern == "uniform"
    assert surface_only.surface_pattern == "latitude"

    # Ocean toggle
    assert baseline.ocean_pattern == "uniform"
    assert ocean_only.ocean_pattern == source.ocean_pattern
    assert ocean_only.resolved_q_star() == source.resolved_q_star()

    # Strain toggle
    assert baseline.epsilon_pole == baseline.epsilon_eq
    assert strain_only.epsilon_pole == source.epsilon_pole
    assert strain_only.tidal_pattern == source.tidal_pattern

    # Shared anchors preserved
    for case in cases.values():
        assert case.T_eq == source.T_eq
        assert case.T_floor == source.T_floor
        assert case.q_ocean_mean == source.q_ocean_mean
