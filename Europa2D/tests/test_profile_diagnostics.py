import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import src  # triggers import path setup

import numpy as np
import pytest

from latitude_profile import LatitudeProfile
from profile_diagnostics import (
    HIGH_LAT_BAND,
    LOW_LAT_BAND,
    area_weighted_band_mean,
    compute_profile_diagnostics,
    ocean_pattern_metadata,
)


def test_area_weighted_band_mean_preserves_constant_profile():
    lats = np.linspace(0.0, 90.0, 37)
    thickness = np.full_like(lats, 20.0)
    assert area_weighted_band_mean(lats, thickness, LOW_LAT_BAND) == pytest.approx(20.0)
    assert area_weighted_band_mean(lats, thickness, HIGH_LAT_BAND) == pytest.approx(20.0)


def test_compute_profile_diagnostics_reports_minimum_latitude():
    lats = np.linspace(0.0, 90.0, 7)
    thickness = np.array([30.0, 24.0, 20.0, 18.0, 19.0, 21.0, 23.0])
    profile = LatitudeProfile(q_ocean_mean=0.02, ocean_pattern="equator_enhanced")
    diagnostics = compute_profile_diagnostics(lats, thickness, profile)
    assert diagnostics.min_thickness_km == pytest.approx(18.0)
    assert diagnostics.min_latitude_deg == pytest.approx(45.0)


def test_ocean_pattern_metadata_uses_correct_primary_sources():
    equator_profile = LatitudeProfile(ocean_pattern="equator_enhanced")
    polar_profile = LatitudeProfile(ocean_pattern="polar_enhanced")
    uniform_profile = LatitudeProfile(ocean_pattern="uniform")

    assert ocean_pattern_metadata(equator_profile).citation == "Soderlund et al. (2014)"
    assert ocean_pattern_metadata(polar_profile).citation == "Lemasquerier et al. (2023)"
    assert ocean_pattern_metadata(uniform_profile).citation == "Ashkenazy & Tziperman (2021)"


def test_equator_enhanced_metadata_reports_ratio_above_one():
    """The equator-enhanced summary should show q_eq/q_pole > 1, not the inverse."""
    profile = LatitudeProfile(
        ocean_pattern="equator_enhanced", q_star=0.4, q_ocean_mean=0.02,
    )
    metadata = ocean_pattern_metadata(profile)
    # ratio = max(q_eq, q_pole) / min(q_eq, q_pole) = q_eq/q_pole for equator_enhanced
    # With q_star=0.4: a = 3*0.4/(3-0.8) = 0.545, ratio = (1+0.545)/(1/(1+2*0.545/3)) ...
    # Just check the summary string contains a number > 1
    assert "q_eq/q_pole" in metadata.summary
    # Extract the number after "q_eq/q_pole = "
    import re
    match = re.search(r"q_eq/q_pole = (\d+\.\d+)", metadata.summary)
    assert match is not None, f"Could not find ratio in: {metadata.summary}"
    reported_ratio = float(match.group(1))
    assert reported_ratio > 1.0, f"Expected ratio > 1, got {reported_ratio}"
