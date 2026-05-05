import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import src

import numpy as np
import pytest
from monte_carlo_2d import MonteCarloRunner2D, MonteCarloResults2D


class TestMonteCarloRunner2D:

    def test_runs_and_returns_results(self):
        """Smoke test: 10 iterations should complete."""
        runner = MonteCarloRunner2D(
            n_iterations=10, seed=42, n_workers=1,
            n_lat=5, nx=21, use_convection=False,
            max_steps=200, dt=1e11,
        )
        results = runner.run()
        assert isinstance(results, MonteCarloResults2D)
        assert results.n_valid > 0
        assert results.H_profiles.shape[1] == 5  # n_lat
        assert results.ocean_pattern == "uniform"
        # Default: uniform_transport -> ocean_amplitude=0.0, q_star=0.0
        assert results.ocean_amplitude == pytest.approx(0.0, abs=1e-10)
        assert results.T_floor == pytest.approx(46.0)
        assert results.q_star == pytest.approx(0.0, abs=1e-10)
        assert results.mantle_tidal_fraction == pytest.approx(0.5)

    def test_H_profiles_are_reasonable(self):
        runner = MonteCarloRunner2D(
            n_iterations=10, seed=42, n_workers=1,
            n_lat=5, nx=21, use_convection=False,
            max_steps=200, dt=1e11,
        )
        results = runner.run()
        assert np.all(results.H_profiles > 0)
        assert np.all(results.H_profiles < 200)

    def test_statistics_computed(self):
        runner = MonteCarloRunner2D(
            n_iterations=10, seed=42, n_workers=1,
            n_lat=5, nx=21, use_convection=False,
            max_steps=200, dt=1e11,
        )
        results = runner.run()
        assert results.H_median.shape == (5,)
        assert results.H_mean.shape == (5,)
        assert results.H_sigma_low.shape == (5,)
        assert results.H_sigma_high.shape == (5,)


def test_results_default_t_floor_matches_2d_baseline():
    """MonteCarloResults2D default T_floor must match the current 2D baseline."""
    import numpy as np
    from monte_carlo_2d import MonteCarloResults2D

    results = MonteCarloResults2D(
        H_profiles=np.zeros((1, 5)),
        latitudes_deg=np.linspace(0, 90, 5),
        n_iterations=1,
        n_valid=1,
        H_median=np.zeros(5),
        H_mean=np.zeros(5),
        H_sigma_low=np.zeros(5),
        H_sigma_high=np.zeros(5),
        runtime_seconds=0.0,
        ocean_pattern="uniform",
        ocean_amplitude=0.0,
        T_c_median=np.zeros(5),
        T_c_mean=np.zeros(5),
        Ti_median=np.zeros(5),
        Ti_mean=np.zeros(5),
    )
    assert results.T_floor == 46.0, (
        f"Default T_floor={results.T_floor}, expected 46.0 K (Ashkenazy low-Q)"
    )


def test_mc_results_have_d_cond_statistics():
    """MC results must include D_cond, D_conv, and conv_fraction median and percentile bands."""
    from monte_carlo_2d import MonteCarloRunner2D
    import numpy as np

    runner = MonteCarloRunner2D(
        n_iterations=10,
        n_lat=5,
        nx=17,
        n_workers=1,
        seed=42,
        ocean_pattern="uniform",
    )
    results = runner.run()

    # D_cond aggregate statistics must exist and have correct shape
    assert results.D_cond_median is not None, "D_cond_median missing"
    assert results.D_cond_mean is not None, "D_cond_mean missing"
    assert results.D_cond_sigma_low is not None, "D_cond_sigma_low missing"
    assert results.D_cond_sigma_high is not None, "D_cond_sigma_high missing"
    assert results.D_cond_median.shape == (5,)
    # D_cond <= H_total at every latitude
    assert np.all(results.D_cond_median <= results.H_median + 0.01)

    # D_conv aggregate statistics
    assert results.D_conv_median is not None, "D_conv_median missing"
    assert results.D_conv_mean is not None, "D_conv_mean missing"
    assert results.D_conv_median.shape == (5,)
    # D_cond + D_conv ≈ H_total per sample (within 0.1 km tolerance);
    # note: median(D_cond) + median(D_conv) ≠ median(H) in general, so we
    # verify using per-sample sums which must agree with the H profiles.
    per_sample_sum = results.D_cond_profiles + results.D_conv_profiles
    assert np.allclose(per_sample_sum, results.H_profiles, atol=0.5), (
        "D_cond + D_conv must equal H_total per sample (within 0.5 km)"
    )

    # Convective fraction aggregate statistics
    assert results.conv_fraction_median is not None, "conv_fraction_median missing"
    assert results.conv_fraction_mean is not None, "conv_fraction_mean missing"
    assert results.conv_fraction_median.shape == (5,)
    assert np.all(results.conv_fraction_median >= 0.0)
    assert np.all(results.conv_fraction_median <= 1.0)


def test_mc_results_have_band_means():
    """MC results must include area-weighted band-mean distributions."""
    from monte_carlo_2d import MonteCarloRunner2D
    import numpy as np

    runner = MonteCarloRunner2D(
        n_iterations=10,
        n_lat=19,
        nx=17,
        n_workers=1,
        seed=42,
        ocean_pattern="uniform",
    )
    results = runner.run()

    # Band-mean distributions: one value per valid MC sample
    assert results.H_low_band is not None, "H_low_band missing"
    assert results.H_high_band is not None, "H_high_band missing"
    assert results.D_cond_low_band is not None, "D_cond_low_band missing"
    assert results.D_cond_high_band is not None, "D_cond_high_band missing"
    assert results.H_low_band.shape == (results.n_valid,)
    assert results.H_high_band.shape == (results.n_valid,)
    # Low-latitude band mean should be finite and positive
    assert np.all(np.isfinite(results.H_low_band))
    assert np.all(results.H_low_band > 0)
