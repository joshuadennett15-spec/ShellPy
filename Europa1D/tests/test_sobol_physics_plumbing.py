"""Verify that Sobol physics configs produce different 1D solver results.

The physics-selector attributes (nu_scaling, conductivity_model, creep_model,
grain_mode) are attached as dynamic attributes to SolverConfig by
run_sobol_analysis._build_solver_config().  This test confirms that
sobol_workflow._evaluate_fixed_params() actually reads them, so different
physics configs produce different equilibrium thicknesses.
"""

import os
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sobol_workflow import _evaluate_fixed_params
from Monte_Carlo import SolverConfig


# Baseline solver params that produce a convecting shell.
SOLVER_PARAMS = {
    "T_surf": 100.0,
    "D_H2O": 120e3,
    "H_rad": 3.5e-12,
    "P_tidal": 2.0e11,
}


def _make_config(nu_scaling=None, conductivity_model=None, creep_model=None, grain_mode=None):
    """Create a lightweight SolverConfig with optional physics overrides."""
    config = SolverConfig()
    config.nx = 21
    config.max_steps = 200
    config.dt = 1e12
    if nu_scaling is not None:
        config.nu_scaling = nu_scaling
    if conductivity_model is not None:
        config.conductivity_model = conductivity_model
    if creep_model is not None:
        config.creep_model = creep_model
    if grain_mode is not None:
        config.grain_mode = grain_mode
    return config


def test_nu_scaling_affects_thickness():
    """Different nu_scaling values must produce different equilibrium thicknesses."""
    config_green = _make_config(nu_scaling="green")
    config_dv = _make_config(nu_scaling="dv2021")

    result_green = _evaluate_fixed_params(
        SOLVER_PARAMS, config_green, physical_output_policy="keep"
    )
    result_dv = _evaluate_fixed_params(
        SOLVER_PARAMS, config_dv, physical_output_policy="keep"
    )

    h_green = result_green["thickness_km"]
    h_dv = result_dv["thickness_km"]

    assert h_green > 0.0, f"Green thickness should be positive, got {h_green}"
    assert h_dv > 0.0, f"DV2021 thickness should be positive, got {h_dv}"
    assert h_green != h_dv, (
        f"nu_scaling should change the result but both gave {h_green:.3f} km. "
        "Physics overrides are likely not being applied."
    )


def test_no_physics_attrs_backward_compatible():
    """A plain SolverConfig (no physics attrs) should still work."""
    config = SolverConfig()
    config.nx = 21
    config.max_steps = 200
    config.dt = 1e12

    result = _evaluate_fixed_params(
        SOLVER_PARAMS, config, physical_output_policy="keep"
    )
    assert result["thickness_km"] > 0.0
    assert result["numerical_success"] == 1.0
