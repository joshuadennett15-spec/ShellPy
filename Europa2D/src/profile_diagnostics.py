"""
Literature-aware diagnostics for 2D latitude profiles and plots.

The helpers in this module keep the plotting scripts focused on
presentation while centralizing the interpretation rules that make the
outputs easier to compare with the Europa ocean-transport literature.
"""
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from latitude_profile import LatitudeProfile


LOW_LAT_BAND = (0.0, 10.0)
HIGH_LAT_BAND = (80.0, 90.0)


@dataclass(frozen=True)
class OceanPatternMetadata:
    """Literature context for a latitude-only ocean forcing proxy."""

    title: str
    citation: str
    reference_url: str
    summary: str
    caution: str


@dataclass(frozen=True)
class ProfileDiagnostics:
    """Compact summary metrics for a single thickness profile."""

    low_band_mean_km: float
    high_band_mean_km: float
    high_minus_low_km: float
    min_thickness_km: float
    min_latitude_deg: float
    max_thickness_km: float
    max_latitude_deg: float
    normalized_contrast: float
    pole_endpoint_km: float
    pole_endpoint_jump_km: float
    q_eq_w_m2: float
    q_pole_w_m2: float
    q_ratio_pole_over_eq: float
    ts_eq_k: float
    ts_pole_k: float
    epsilon_eq: float
    epsilon_pole: float
    mean_nu: float
    max_nu: float
    convective_fraction: float
    q_star: float
    mantle_tidal_fraction: float
    T_floor: float


def ocean_pattern_metadata(profile: LatitudeProfile) -> OceanPatternMetadata:
    """Return literature metadata for the selected ocean forcing proxy."""
    ratio = profile.ocean_endpoint_ratio()
    if profile.ocean_pattern == "uniform":
        return OceanPatternMetadata(
            title="Uniform transport proxy",
            citation="Ashkenazy & Tziperman (2021)",
            reference_url="https://www.nature.com/articles/s41467-021-26710-0",
            summary="Represents efficient meridional redistribution and a near-uniform shell benchmark.",
            caution="This is a transport-limit proxy, not a resolved salinity-coupled ocean simulation.",
        )
    if profile.ocean_pattern == "equator_enhanced":
        return OceanPatternMetadata(
            title="Equator-enhanced proxy",
            citation="Soderlund et al. (2014)",
            reference_url="https://doi.org/10.1038/ngeo2021",
            summary=(
                f"Uses a latitude-only proxy with q_eq/q_pole = {ratio:.2f}, "
                "matching the zonal-mean low-latitude enhancement reported in the paper."
            ),
            caution="Longitude structure and time variability from the 3D ocean model are not represented.",
        )
    if profile.ocean_pattern == "polar_enhanced":
        return OceanPatternMetadata(
            title="Polar-enhanced proxy",
            citation="Lemasquerier et al. (2023)",
            reference_url="https://doi.org/10.1029/2023AV000994",
            summary=(
                f"Uses a conservative latitude-only proxy with q_pole/q_eq = {ratio:.2f}, "
                "representing mantle-tidal cases where polar flux into the ice is enhanced."
            ),
            caution="This is a zonally averaged proxy; the paper's full pattern also contains longitude dependence.",
        )
    raise ValueError(f"Unknown ocean pattern: {profile.ocean_pattern}")


def area_weighted_band_mean(
    latitudes_deg: npt.ArrayLike,
    values: npt.ArrayLike,
    band: tuple[float, float],
) -> float:
    """
    Compute an area-weighted mean over a latitude band.

    Geographic latitude bands are weighted by cos(phi), which is the
    hemisphere-area Jacobian in axisymmetric coordinates.
    """
    latitudes = np.asarray(latitudes_deg, dtype=float)
    values_arr = np.asarray(values, dtype=float)
    mask = (latitudes >= band[0]) & (latitudes <= band[1])
    if not np.any(mask):
        raise ValueError(f"No points available in latitude band {band}.")
    weights = np.cos(np.radians(latitudes[mask]))
    return float(np.average(values_arr[mask], weights=weights))


def band_mean_samples(
    latitudes_deg: npt.ArrayLike,
    profiles: npt.ArrayLike,
    band: tuple[float, float],
) -> npt.NDArray[np.float64]:
    """Compute area-weighted band means for each row in a profile matrix."""
    latitudes = np.asarray(latitudes_deg, dtype=float)
    profiles_arr = np.asarray(profiles, dtype=float)
    if profiles_arr.ndim != 2:
        raise ValueError("profiles must have shape (n_samples, n_lat).")
    mask = (latitudes >= band[0]) & (latitudes <= band[1])
    if not np.any(mask):
        raise ValueError(f"No points available in latitude band {band}.")
    weights = np.cos(np.radians(latitudes[mask]))
    return np.average(profiles_arr[:, mask], axis=1, weights=weights)


def compute_profile_diagnostics(
    latitudes_deg: npt.ArrayLike,
    thickness_km: npt.ArrayLike,
    profile: LatitudeProfile,
    nu_profile: npt.ArrayLike | None = None,
) -> ProfileDiagnostics:
    """Compute summary diagnostics for a single 2D thickness profile."""
    latitudes = np.asarray(latitudes_deg, dtype=float)
    thickness = np.asarray(thickness_km, dtype=float)

    low_band_mean = area_weighted_band_mean(latitudes, thickness, LOW_LAT_BAND)
    high_band_mean = area_weighted_band_mean(latitudes, thickness, HIGH_LAT_BAND)

    min_index = int(np.argmin(thickness))
    max_index = int(np.argmax(thickness))

    q_eq, q_pole = profile.ocean_endpoint_fluxes()
    ts_eq = float(profile.surface_temperature(0.0))
    ts_pole = float(profile.surface_temperature(np.pi / 2))
    eps_eq = float(profile.tidal_strain(0.0))
    eps_pole = float(profile.tidal_strain(np.pi / 2))

    if nu_profile is None:
        nu = np.ones_like(thickness)
    else:
        nu = np.asarray(nu_profile, dtype=float)

    return ProfileDiagnostics(
        low_band_mean_km=low_band_mean,
        high_band_mean_km=high_band_mean,
        high_minus_low_km=high_band_mean - low_band_mean,
        min_thickness_km=float(thickness[min_index]),
        min_latitude_deg=float(latitudes[min_index]),
        max_thickness_km=float(thickness[max_index]),
        max_latitude_deg=float(latitudes[max_index]),
        normalized_contrast=float((thickness.max() - thickness.min()) / thickness.mean()),
        pole_endpoint_km=float(thickness[-1]),
        pole_endpoint_jump_km=float(thickness[-1] - thickness[-2]) if thickness.size > 1 else 0.0,
        q_eq_w_m2=q_eq,
        q_pole_w_m2=q_pole,
        q_ratio_pole_over_eq=np.inf if q_eq == 0.0 else q_pole / q_eq,
        ts_eq_k=ts_eq,
        ts_pole_k=ts_pole,
        epsilon_eq=eps_eq,
        epsilon_pole=eps_pole,
        mean_nu=float(np.mean(nu)),
        max_nu=float(np.max(nu)),
        convective_fraction=float(np.mean(nu > 1.01)),
        q_star=profile.resolved_q_star(),
        mantle_tidal_fraction=profile.mantle_tidal_fraction,
        T_floor=profile.T_floor,
    )


def format_diagnostic_lines(
    metadata: OceanPatternMetadata,
    diagnostics: ProfileDiagnostics,
) -> list[str]:
    """Create concise annotation lines for science-oriented figures."""
    return [
        f"{metadata.title} ({metadata.citation})",
        f"H(0-10 deg) = {diagnostics.low_band_mean_km:.2f} km",
        f"H(80-90 deg) = {diagnostics.high_band_mean_km:.2f} km",
        f"Delta H_high-low = {diagnostics.high_minus_low_km:+.2f} km",
        f"H_min = {diagnostics.min_thickness_km:.2f} km at {diagnostics.min_latitude_deg:.1f} deg",
        f"q_pole/q_eq = {diagnostics.q_ratio_pole_over_eq:.2f}",
        f"q* = {diagnostics.q_star:.3f} (Lemasquerier 2023)",
        f"mantle tidal fraction = {diagnostics.mantle_tidal_fraction:.2f}",
        f"T_floor = {diagnostics.T_floor:.1f} K",
        f"max Nu = {diagnostics.max_nu:.2f}",
        "Interpret 90 deg as a symmetry boundary node, not an interior latitude.",
    ]
