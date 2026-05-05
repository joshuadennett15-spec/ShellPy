"""Tests for config-driven conductivity model selection (Task 2)."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
from constants import Thermal
from ConfigManager import ConfigManager


def test_carnahan_conductivity_at_100K():
    """Carnahan et al. (2021, EPSL 563): k(T) = 612/T."""
    k = Thermal.conductivity(100.0, model="Carnahan")
    assert k == pytest.approx(6.12, rel=1e-6)


def test_howell_conductivity_at_100K():
    """Klinger 1980 / Howell 2021: k(T) = 567/T."""
    k = Thermal.conductivity(100.0, model="Howell")
    assert k == pytest.approx(5.67, rel=1e-6)


def test_default_is_carnahan():
    """Default conductivity model should be Carnahan after config update."""
    model = ConfigManager.get("thermal", "CONDUCTIVITY_MODEL", "Carnahan")
    assert model == "Carnahan"


def test_default_conductivity_uses_config():
    """When model=None, conductivity() should read from config and use Carnahan."""
    k = Thermal.conductivity(100.0)
    assert k == pytest.approx(6.12, rel=1e-6)
