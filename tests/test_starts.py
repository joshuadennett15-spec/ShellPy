"""With DV2021 applied to the whole shell, where a column starts does not decide where it ends,
except within a few tens of metres to a kilometre at the convection onset."""
import pytest

from shellpy.monte_carlo import sample_1d, solve_1d

from cases import CONDUCTIVE, CONVECTING_OCEAN


@pytest.mark.parametrize("p", [CONDUCTIVE, CONVECTING_OCEAN])
def test_both_starts_reach_the_same_equilibrium(p):
    thin, thick = (solve_1d(p, start=s) for s in ('thin', 'thick'))
    assert thin['status'] == thick['status'] == 'ok'
    assert thin['H_km'] == pytest.approx(thick['H_km'], abs=0.01)


def test_column_cycling_at_convection_onset_converges():
    out = solve_1d(sample_1d(168, seed=10042), start='thick')   # switches convection on and off
    assert out['status'] == 'ok'
    assert 0.01 < out['H_spread_km'] < 0.2
