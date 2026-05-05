import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pytest
from audited_equatorial_sampler import AuditedEquatorialSampler
from constants import Planetary


class TestEquatorialOverrides:
    def test_T_surf_equatorial_range(self):
        sampler = AuditedEquatorialSampler(seed=42, enhancement_factor=1.0)
        t_surfs = [sampler.sample()['T_surf'] for _ in range(500)]
        assert all(90.0 <= t <= 104.0 for t in t_surfs)
        mean_t = np.mean(t_surfs)
        assert 92.0 < mean_t < 100.0

    def test_epsilon_0_equatorial_range(self):
        sampler = AuditedEquatorialSampler(seed=42, enhancement_factor=1.0)
        epsilons = [sampler.sample()['epsilon_0'] for _ in range(500)]
        assert all(3e-6 <= e <= 1.2e-5 for e in epsilons)
        log_mean = np.mean(np.log10(epsilons))
        assert np.log10(4e-6) < log_mean < np.log10(1e-5)

    def test_d_grain_in_audited_range(self):
        sampler = AuditedEquatorialSampler(seed=42, enhancement_factor=1.0)
        grains = [sampler.sample()['d_grain'] for _ in range(100)]
        assert all(5e-5 <= d <= 4e-3 for d in grains)


class TestTidalEnhancement:
    def _reconstruct_q_basal(self, params):
        H_rad = params['H_rad']
        D_H2O = params['D_H2O']
        R_rock = Planetary.RADIUS - D_H2O
        M_rock = (4.0 / 3.0) * np.pi * (R_rock ** 3) * 3500.0
        q_rad = (H_rad * M_rock) / Planetary.AREA
        q_tidal = params['P_tidal'] / Planetary.AREA
        return q_rad + q_tidal, q_rad, q_tidal

    def test_baseline_no_scaling(self):
        sampler = AuditedEquatorialSampler(seed=42, enhancement_factor=1.0)
        params = sampler.sample()
        q_basal, _, _ = self._reconstruct_q_basal(params)
        assert 0.005 <= q_basal <= 0.025

    def test_moderate_increases_or_matches_q_basal(self):
        rng_seed = 99
        s1 = AuditedEquatorialSampler(seed=rng_seed, enhancement_factor=1.0)
        p1 = s1.sample()
        s2 = AuditedEquatorialSampler(seed=rng_seed, enhancement_factor=1.2)
        p2 = s2.sample()
        q1, _, q_tid1 = self._reconstruct_q_basal(p1)
        q2, _, q_tid2 = self._reconstruct_q_basal(p2)
        if q_tid1 > 0:
            assert q2 > q1
        else:
            np.testing.assert_allclose(q1, q2, rtol=1e-10)

    def test_only_tidal_component_scales(self):
        rng_seed = 77
        s1 = AuditedEquatorialSampler(seed=rng_seed, enhancement_factor=1.0)
        p1 = s1.sample()
        s2 = AuditedEquatorialSampler(seed=rng_seed, enhancement_factor=1.5)
        p2 = s2.sample()
        q1, q_rad1, q_tid1 = self._reconstruct_q_basal(p1)
        q2, q_rad2, q_tid2 = self._reconstruct_q_basal(p2)
        np.testing.assert_allclose(q_rad1, q_rad2, rtol=1e-10)
        np.testing.assert_allclose(q_tid2, 1.5 * q_tid1, rtol=1e-10)

    def test_enhancement_factor_stored_in_params(self):
        sampler = AuditedEquatorialSampler(seed=42, enhancement_factor=1.2)
        params = sampler.sample()
        assert params.get('eq_enhancement') == pytest.approx(1.2)


class TestReproducibility:
    def test_same_seed_same_output(self):
        s1 = AuditedEquatorialSampler(seed=42, enhancement_factor=1.0)
        s2 = AuditedEquatorialSampler(seed=42, enhancement_factor=1.0)
        p1 = s1.sample()
        p2 = s2.sample()
        assert p1['T_surf'] == p2['T_surf']
        assert p1['epsilon_0'] == p2['epsilon_0']
        assert p1['P_tidal'] == p2['P_tidal']
