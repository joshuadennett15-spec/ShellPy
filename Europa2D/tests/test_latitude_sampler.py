import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import src

import pytest

from latitude_sampler import LatitudeParameterSampler
from latitude_profile import LatitudeProfile


class TestLatitudeParameterSampler:

    def test_returns_dict_and_profile(self):
        sampler = LatitudeParameterSampler(seed=42)
        params, profile = sampler.sample()
        assert isinstance(params, dict)
        assert isinstance(profile, LatitudeProfile)

    def test_shared_params_present(self):
        sampler = LatitudeParameterSampler(seed=42)
        params, _ = sampler.sample()
        required = [
            'd_grain', 'Q_v', 'Q_b', 'mu_ice', 'D0v', 'D0b', 'd_del',
            'f_porosity', 'f_salt', 'B_k', 'T_phi', 'H_rad',
            'q_basal', 'q_tidal',
        ]
        for key in required:
            assert key in params, f"Missing key: {key}"

    def test_reproducible_with_seed(self):
        s1 = LatitudeParameterSampler(seed=42)
        s2 = LatitudeParameterSampler(seed=42)
        p1, prof1 = s1.sample()
        p2, prof2 = s2.sample()
        assert p1['d_grain'] == p2['d_grain']
        assert prof1.T_eq == prof2.T_eq

    def test_profile_has_valid_values(self):
        sampler = LatitudeParameterSampler(seed=42)
        _, profile = sampler.sample()
        assert 80 <= profile.T_eq <= 120
        assert 3e-6 <= profile.epsilon_eq <= 1.2e-5
        assert 6e-6 <= profile.epsilon_pole <= 2.5e-5
        assert profile.q_ocean_mean > 0

    def test_audited_fixed_params(self):
        """Audited priors: f_salt=0, B_k=1, T_phi=150."""
        sampler = LatitudeParameterSampler(seed=42)
        params, _ = sampler.sample()
        assert params['f_salt'] == 0.0
        assert params['B_k'] == 1.0
        assert params['T_phi'] == 150.0

    def test_2d_q_basal_range_with_modest_tidal_uplift(self):
        """2D q_basal should stay close to the audited range after the uplift."""
        sampler = LatitudeParameterSampler(seed=0)
        q_vals = [sampler.sample()[0]['q_basal'] for _ in range(200)]
        assert all(4e-3 <= q <= 35e-3 for q in q_vals)

    def test_inherited_q_basal_is_preserved_for_diagnostics(self):
        """Store the pre-uplift audited q_basal so runs remain comparable."""
        sampler = LatitudeParameterSampler(seed=0)
        q_vals = [sampler.sample()[0]['q_basal_inherited'] for _ in range(200)]
        assert all(4e-3 <= q <= 30e-3 for q in q_vals)

    def test_audited_H_rad_positive(self):
        """H_rad must be truncated > 0."""
        sampler = LatitudeParameterSampler(seed=0)
        for _ in range(200):
            params, _ = sampler.sample()
            assert params['H_rad'] > 0

    def test_audited_f_porosity_range(self):
        """f_porosity tightened to [0, 0.10]."""
        sampler = LatitudeParameterSampler(seed=0)
        for _ in range(200):
            params, _ = sampler.sample()
            assert 0.0 <= params['f_porosity'] <= 0.10

    def test_audited_d_grain_range(self):
        """d_grain should match the audited 0.05-4.0 mm bounds."""
        sampler = LatitudeParameterSampler(seed=0)
        for _ in range(200):
            params, _ = sampler.sample()
            assert 5e-5 <= params['d_grain'] <= 4e-3

    def test_shared_vs_latitude_parameter_partition_is_explicit(self):
        shared = LatitudeParameterSampler.shared_parameter_names()
        latitude = LatitudeParameterSampler.latitude_structure_names()

        assert 'd_grain' in shared
        assert 'D_H2O' in shared
        assert 'T_eq' not in shared
        assert 'epsilon_eq' in latitude
        assert 'q_ocean_mean' in latitude
        assert set(shared).isdisjoint(latitude)


class TestSamplerNewFields:

    def test_profile_has_T_floor(self):
        sampler = LatitudeParameterSampler(seed=42)
        _, profile = sampler.sample()
        assert hasattr(profile, 'T_floor')
        assert 42 <= profile.T_floor <= 59

    def test_profile_has_mantle_tidal_fraction(self):
        sampler = LatitudeParameterSampler(seed=42)
        _, profile = sampler.sample()
        assert hasattr(profile, 'mantle_tidal_fraction')
        assert 0.0 < profile.mantle_tidal_fraction < 1.0

    def test_T_floor_independent_of_q_ocean(self):
        """T_floor should not be derived from q_ocean_mean."""
        sampler = LatitudeParameterSampler(seed=42)
        _, profile = sampler.sample()
        assert profile.T_floor != pytest.approx(52.0 + 240.0 * profile.q_ocean_mean, rel=0.01)

    def test_T_floor_less_than_T_eq(self):
        """Guard: T_floor must always be < T_eq."""
        sampler = LatitudeParameterSampler(seed=0)
        for _ in range(200):
            _, profile = sampler.sample()
            assert profile.T_floor < profile.T_eq

    def test_T_floor_varies_across_samples(self):
        """T_floor should be sampled, not a constant default."""
        sampler = LatitudeParameterSampler(seed=0)
        floors = [sampler.sample()[1].T_floor for _ in range(50)]
        assert len(set(floors)) > 1

    def test_mantle_tidal_fraction_varies_across_samples(self):
        """mantle_tidal_fraction should be sampled, not a constant default."""
        sampler = LatitudeParameterSampler(seed=0)
        fracs = [sampler.sample()[1].mantle_tidal_fraction for _ in range(50)]
        assert len(set(fracs)) > 1

    def test_mantle_tidal_fraction_range(self):
        """mantle_tidal_fraction ~ Uniform(0.1, 0.9)."""
        sampler = LatitudeParameterSampler(seed=0)
        for _ in range(200):
            _, profile = sampler.sample()
            assert 0.1 <= profile.mantle_tidal_fraction <= 0.9

    def test_equator_enhanced_has_explicit_q_star(self):
        """For equator_enhanced pattern, q_star should be sampled directly."""
        sampler = LatitudeParameterSampler(seed=42, ocean_pattern="equator_enhanced")
        _, profile = sampler.sample()
        assert profile.q_star is not None
        assert 0.1 <= profile.q_star <= 0.8

    def test_polar_enhanced_has_no_explicit_q_star(self):
        """For polar_enhanced, q_star should be None (derived from mantle_tidal_fraction)."""
        sampler = LatitudeParameterSampler(seed=42, ocean_pattern="polar_enhanced")
        _, profile = sampler.sample()
        assert profile.q_star is None

    def test_sampler_records_tidal_scale_and_uplift(self):
        sampler = LatitudeParameterSampler(seed=42)
        params, _ = sampler.sample()
        assert params["q_tidal_scale"] == pytest.approx(1.0)
        assert params["q_basal"] >= params["q_basal_inherited"]


def test_sampler_passes_tidal_pattern():
    """Sampler propagates tidal_pattern to LatitudeProfile."""
    sampler = LatitudeParameterSampler(seed=42, tidal_pattern="shell_dominated")
    _, profile = sampler.sample()
    assert profile.tidal_pattern == "shell_dominated"
