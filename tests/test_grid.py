"""The equilibrium must not depend on the grid: halving the node spacing barely moves a convecting shell."""
import pytest

from shellpy.monte_carlo import basal_heat_flux, run_column

from cases import CONVECTING


def test_convecting_thickness_is_grid_converged():
    q = basal_heat_flux(CONVECTING)
    coarse, fine = (run_column(CONVECTING, q, 25e3, nx=nx) for nx in (121, 241))
    assert coarse.converged and fine.converged
    assert coarse.state.is_convecting and fine.state.is_convecting
    assert coarse.H == pytest.approx(fine.H, rel=0.005)   # 31 nodes were 2.7% off
