import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pytest
from audited_endmember_sampler import AuditedEndmemberSampler
from constants import Planetary


# Endmember presets (from spec)
EQ_PRESET = dict(
    T_surf_mean=110.0, T_surf_std=5.0, T_surf_clip=(95.0, 120.0),
    epsilon_0_log_center=np.log10(6e-6), epsilon_0_log_sigma=0.2,
    epsilon_0_clip=(2e-6, 2e-5),
)
POLE_PRESET = dict(
    T_surf_mean=50.0, T_surf_std=5.0, T_surf_clip=(45.0, 80.0),
    epsilon_0_log_center=np.log10(1.2e-5), epsilon_0_log_sigma=0.2,
    epsilon_0_clip=(2e-6, 3.4e-5),
)


def _reconstruct_q_tidal(params):
    """Recover q_tidal from P_tidal for test assertions."""
    return params['P_tidal'] / Planetary.AREA


class TestEndmemberOverrides:
    def test_eq_T_surf_in_range(self):
        sampler = AuditedEndmemberSampler(seed=42, q_tidal_multiplier=1.0, **EQ_PRESET)
        t_surfs = [sampler.sample()['T_surf'] for _ in range(500)]
        assert all(95.0 <= t <= 120.0 for t in t_surfs)
        assert 105.0 < np.mean(t_surfs) < 115.0

    def test_pole_T_surf_in_range(self):
        sampler = AuditedEndmemberSampler(seed=42, q_tidal_multiplier=1.0, **POLE_PRESET)
        t_surfs = [sampler.sample()['T_surf'] for _ in range(500)]
        assert all(45.0 <= t <= 80.0 for t in t_surfs)
        assert 45.0 < np.mean(t_surfs) < 58.0

    def test_eq_epsilon_0_in_range(self):
        sampler = AuditedEndmemberSampler(seed=42, q_tidal_multiplier=1.0, **EQ_PRESET)
        epsilons = [sampler.sample()['epsilon_0'] for _ in range(500)]
        assert all(2e-6 <= e <= 2e-5 for e in epsilons)
        log_mean = np.mean(np.log10(epsilons))
        assert np.log10(4e-6) < log_mean < np.log10(1e-5)

    def test_pole_epsilon_0_in_range(self):
        sampler = AuditedEndmemberSampler(seed=42, q_tidal_multiplier=1.0, **POLE_PRESET)
        epsilons = [sampler.sample()['epsilon_0'] for _ in range(500)]
        assert all(2e-6 <= e <= 3.4e-5 for e in epsilons)
        log_mean = np.mean(np.log10(epsilons))
        assert np.log10(8e-6) < log_mean < np.log10(2e-5)

    def test_d_grain_not_overridden(self):
        sampler = AuditedEndmemberSampler(seed=42, q_tidal_multiplier=1.0, **EQ_PRESET)
        grains = [sampler.sample()['d_grain'] for _ in range(100)]
        assert all(5e-5 <= d <= 4e-3 for d in grains)


class TestQtidalMultiplier:
    def test_multiplier_1_no_change(self):
        from audited_sampler import AuditedShellSampler
        base = AuditedShellSampler(seed=42)
        endm = AuditedEndmemberSampler(seed=42, q_tidal_multiplier=1.0, **EQ_PRESET)
        p_base = base.sample()
        p_endm = endm.sample()
        # P_tidal should be identical (multiplier=1.0 doesn't change it)
        np.testing.assert_allclose(p_endm['P_tidal'], p_base['P_tidal'], rtol=1e-10)

    def test_multiplier_scales_P_tidal(self):
        s1 = AuditedEndmemberSampler(seed=42, q_tidal_multiplier=1.0, **EQ_PRESET)
        s2 = AuditedEndmemberSampler(seed=42, q_tidal_multiplier=1.15, **EQ_PRESET)
        p1 = s1.sample()
        p2 = s2.sample()
        np.testing.assert_allclose(p2['P_tidal'], 1.15 * p1['P_tidal'], rtol=1e-10)

    def test_multiplier_less_than_one(self):
        s1 = AuditedEndmemberSampler(seed=42, q_tidal_multiplier=1.0, **POLE_PRESET)
        s2 = AuditedEndmemberSampler(seed=42, q_tidal_multiplier=0.70, **POLE_PRESET)
        p1 = s1.sample()
        p2 = s2.sample()
        np.testing.assert_allclose(p2['P_tidal'], 0.70 * p1['P_tidal'], rtol=1e-10)

    def test_radiogenic_unchanged_by_multiplier(self):
        s1 = AuditedEndmemberSampler(seed=42, q_tidal_multiplier=1.0, **EQ_PRESET)
        s2 = AuditedEndmemberSampler(seed=42, q_tidal_multiplier=1.30, **EQ_PRESET)
        p1 = s1.sample()
        p2 = s2.sample()
        # H_rad and D_H2O are drawn before overrides, should be identical
        assert p1['H_rad'] == p2['H_rad']
        assert p1['D_H2O'] == p2['D_H2O']

    def test_multiplier_stored_in_params(self):
        sampler = AuditedEndmemberSampler(seed=42, q_tidal_multiplier=0.85, **EQ_PRESET)
        params = sampler.sample()
        assert params.get('q_tidal_multiplier') == pytest.approx(0.85)


class TestReproducibility:
    def test_same_seed_same_output(self):
        s1 = AuditedEndmemberSampler(seed=42, q_tidal_multiplier=1.0, **EQ_PRESET)
        s2 = AuditedEndmemberSampler(seed=42, q_tidal_multiplier=1.0, **EQ_PRESET)
        p1 = s1.sample()
        p2 = s2.sample()
        assert p1['T_surf'] == p2['T_surf']
        assert p1['epsilon_0'] == p2['epsilon_0']
        assert p1['P_tidal'] == p2['P_tidal']


import pickle


class TestNamedSubclasses:
    """Verify each named subclass is picklable and produces correct overrides."""

    SUBCLASS_CONFIGS = [
        ("UniformEqSampler",        110.0, (95.0, 120.0), 6e-6,  (2e-6, 2e-5),   1.00),
        ("UniformPoleSampler",       50.0, (45.0, 80.0),  1.2e-5,(2e-6, 3.4e-5), 1.00),
        ("SoderlundEqSampler",      110.0, (95.0, 120.0), 6e-6,  (2e-6, 2e-5),   1.15),
        ("SoderlundPoleSampler",     50.0, (45.0, 80.0),  1.2e-5,(2e-6, 3.4e-5), 0.70),
        ("LemasquerierEqSampler",   110.0, (95.0, 120.0), 6e-6,  (2e-6, 2e-5),   0.85),
        ("LemasquerierPoleSampler",  50.0, (45.0, 80.0),  1.2e-5,(2e-6, 3.4e-5), 1.30),
    ]

    @pytest.mark.parametrize("cls_name,t_mean,t_clip,eps_center,eps_clip,mult",
                             SUBCLASS_CONFIGS)
    def test_subclass_is_picklable(self, cls_name, t_mean, t_clip, eps_center, eps_clip, mult):
        import audited_endmember_sampler as mod
        cls = getattr(mod, cls_name)
        sampler = cls(seed=42)
        roundtripped = pickle.loads(pickle.dumps(sampler))
        p1 = sampler.sample()
        # Re-create (pickle roundtrip resets RNG state, so just verify it works)
        p2 = cls(seed=42).sample()
        assert p1['T_surf'] == p2['T_surf']

    @pytest.mark.parametrize("cls_name,t_mean,t_clip,eps_center,eps_clip,mult",
                             SUBCLASS_CONFIGS)
    def test_subclass_T_surf_in_range(self, cls_name, t_mean, t_clip, eps_center, eps_clip, mult):
        import audited_endmember_sampler as mod
        cls = getattr(mod, cls_name)
        sampler = cls(seed=42)
        t_surfs = [sampler.sample()['T_surf'] for _ in range(200)]
        assert all(t_clip[0] <= t <= t_clip[1] for t in t_surfs)

    @pytest.mark.parametrize("cls_name,t_mean,t_clip,eps_center,eps_clip,mult",
                             SUBCLASS_CONFIGS)
    def test_subclass_multiplier_correct(self, cls_name, t_mean, t_clip, eps_center, eps_clip, mult):
        import audited_endmember_sampler as mod
        cls = getattr(mod, cls_name)
        sampler = cls(seed=42)
        params = sampler.sample()
        assert params['q_tidal_multiplier'] == pytest.approx(mult)

    @pytest.mark.parametrize("cls_name,t_mean,t_clip,eps_center,eps_clip,mult",
                             SUBCLASS_CONFIGS)
    def test_subclass_accepts_only_seed(self, cls_name, t_mean, t_clip, eps_center, eps_clip, mult):
        """MonteCarloRunner calls sampler_class(seed=N). Verify this works."""
        import audited_endmember_sampler as mod
        cls = getattr(mod, cls_name)
        sampler = cls(seed=99)
        params = sampler.sample()
        assert 'T_surf' in params
        assert 'P_tidal' in params
