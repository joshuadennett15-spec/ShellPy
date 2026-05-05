"""
Helpers for paired 4-case attribution experiments.

Each realization shares one sampled shell/forcing state, then toggles a single
latitude-varying forcing at a time:
    baseline     : uniform surface, uniform ocean, uniform strain
    surface_only : latitude-varying surface, uniform ocean, uniform strain
    ocean_only   : uniform surface, varying ocean, uniform strain
    strain_only  : uniform surface, uniform ocean, varying strain
"""
from __future__ import annotations

from typing import Final

from latitude_profile import LatitudeProfile


ATTRIBUTION_CASES: Final[tuple[str, ...]] = (
    "baseline",
    "surface_only",
    "ocean_only",
    "strain_only",
)


def build_paired_attribution_profiles(source_profile: LatitudeProfile) -> dict[str, LatitudeProfile]:
    """Build the paired 4-case attribution profiles from one sampled source profile."""
    common_kwargs = dict(
        T_eq=source_profile.T_eq,
        T_floor=source_profile.T_floor,
        epsilon_eq=source_profile.epsilon_eq,
        q_ocean_mean=source_profile.q_ocean_mean,
        mantle_tidal_fraction=source_profile.mantle_tidal_fraction,
        strict_q_star=False,
        mid_latitude_amplification=source_profile.mid_latitude_amplification,
        surface_temp_exponent=source_profile.surface_temp_exponent,
        grain_latitude_mode=source_profile.grain_latitude_mode,
        grain_strain_exponent=source_profile.grain_strain_exponent,
    )

    # Uniformized cases force mantle_core with equal endpoints so the strain
    # field is actually flat even if the sampled campaign uses another family.
    uniform_strain_kwargs = dict(
        epsilon_pole=source_profile.epsilon_eq,
        tidal_pattern="mantle_core",
    )

    ocean_case_kwargs = dict(
        ocean_pattern=source_profile.ocean_pattern,
        ocean_amplitude=source_profile.ocean_amplitude,
        q_star=source_profile.q_star,
    )

    return {
        "baseline": LatitudeProfile(
            **common_kwargs,
            **uniform_strain_kwargs,
            ocean_pattern="uniform",
            surface_pattern="uniform",
        ),
        "surface_only": LatitudeProfile(
            **common_kwargs,
            **uniform_strain_kwargs,
            ocean_pattern="uniform",
            surface_pattern="latitude",
        ),
        "ocean_only": LatitudeProfile(
            **common_kwargs,
            **uniform_strain_kwargs,
            surface_pattern="uniform",
            **ocean_case_kwargs,
        ),
        "strain_only": LatitudeProfile(
            **common_kwargs,
            epsilon_pole=source_profile.epsilon_pole,
            ocean_pattern="uniform",
            q_star=None,
            ocean_amplitude=None,
            tidal_pattern=source_profile.tidal_pattern,
            surface_pattern="uniform",
        ),
    }
