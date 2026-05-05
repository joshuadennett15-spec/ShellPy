"""
Budget-constrained regional parameter samplers for ocean heat transport comparison.

Suite A: clean ocean-model comparison.
Only T_surf and P_tidal vary between equator and pole.
All other parameters (epsilon_0, d_grain, rheology, composition) use the
shared global Howell (2021) distributions.

Three literature-grounded redistribution scenarios split a fixed total
budget P_total between equator and pole:

    Uniform             Ashkenazy & Tziperman (2021)   50 / 50
    Equatorial-enhanced Soderlund et al. (2014)        58.3 / 41.7
    Polar-enhanced      Lemasquerier et al. (2023)     33.3 / 66.7

References
----------
Ashkenazy, Y. & Tziperman, E. (2021). Nature Communications, 12, 6376.
Soderlund, K. M. et al. (2014). Nature Geoscience, 7, 16-19.
Lemasquerier, D. G. et al. (2023). AGU Advances, 4, e2023AV000994.
Howell, S. M. (2021). The likely thickness of Europa's icy shell.
"""
import numpy as np
from typing import Dict, Optional

from Monte_Carlo import HowellParameterSampler


# ── Redistribution fractions ────────────────────────────────────────────────
# Keys: (equator_fraction, pole_fraction)

UNIFORM = (0.500, 0.500)
SODERLUND_2014 = (0.583, 0.417)
LEMASQUERIER_2023_CONSERVATIVE = (0.333, 0.667)
LEMASQUERIER_2023_STRONG = (0.273, 0.727)

# Default total budget (Watts)
DEFAULT_P_TOTAL = 500e9  # 500 GW


# ── Base budget-constrained sampler ─────────────────────────────────────────

class BudgetRegionalSampler(HowellParameterSampler):
    """
    Regional sampler that inherits the full Howell (2021) parameter set
    and overrides only T_surf and P_tidal.

    Parameters
    ----------
    T_surf_mean : float
        Mean surface temperature for this latitude (K).
    T_surf_std : float
        Standard deviation of surface temperature (K).
    T_surf_clip : tuple of float
        (low, high) clipping bounds for T_surf (K).
    p_fraction : float
        Fraction of P_total assigned to this latitude endpoint.
    P_total : float
        Total silicate tidal power budget (W). Default 500 GW.
    seed : int or None
        Random seed.
    """

    def __init__(
        self,
        T_surf_mean: float,
        T_surf_std: float,
        T_surf_clip: tuple,
        p_fraction: float,
        P_total: float = DEFAULT_P_TOTAL,
        seed: Optional[int] = None,
    ):
        super().__init__(seed=seed)
        self._T_surf_mean = T_surf_mean
        self._T_surf_std = T_surf_std
        self._T_surf_clip = T_surf_clip
        self._p_fraction = p_fraction
        self._P_total = P_total

    def sample(self) -> Dict[str, float]:
        params = super().sample()

        # Override T_surf with latitude-specific distribution
        T_surf = self.rng.normal(self._T_surf_mean, self._T_surf_std)
        params["T_surf"] = float(np.clip(T_surf, *self._T_surf_clip))

        # Override P_tidal with budget fraction
        params["P_tidal"] = self._p_fraction * self._P_total

        return params


# ── Surface temperature constants ───────────────────────────────────────────
# Ojakangas & Stevenson (1989)
_EQ_T = (108.0, 2.0, (100.0, 115.0))   # mean, std, (low, high)
_POLE_T = (50.0, 5.0, (35.0, 70.0))


# ── Convenience factory ─────────────────────────────────────────────────────

def make_sampler(
    region: str,
    scenario: str,
    P_total: float = DEFAULT_P_TOTAL,
    seed: Optional[int] = None,
) -> BudgetRegionalSampler:
    """
    Create a budget-constrained sampler for a given region and scenario.

    Parameters
    ----------
    region : {"equator", "pole"}
    scenario : {"uniform", "soderlund2014", "lemasquerier2023",
                "lemasquerier2023_strong"}
    P_total : float
        Total budget in Watts.  Default 500 GW.
    seed : int or None
    """
    fractions = {
        "uniform": UNIFORM,
        "soderlund2014": SODERLUND_2014,
        "lemasquerier2023": LEMASQUERIER_2023_CONSERVATIVE,
        "lemasquerier2023_strong": LEMASQUERIER_2023_STRONG,
    }

    if scenario not in fractions:
        raise ValueError(f"Unknown scenario {scenario!r}. "
                         f"Choose from {list(fractions)}")

    eq_frac, pole_frac = fractions[scenario]

    if region == "equator":
        T_mean, T_std, T_clip = _EQ_T
        frac = eq_frac
    elif region == "pole":
        T_mean, T_std, T_clip = _POLE_T
        frac = pole_frac
    else:
        raise ValueError(f"Unknown region {region!r}. Choose 'equator' or 'pole'")

    return BudgetRegionalSampler(
        T_surf_mean=T_mean,
        T_surf_std=T_std,
        T_surf_clip=T_clip,
        p_fraction=frac,
        P_total=P_total,
        seed=seed,
    )


# ── Named sampler classes (picklable for multiprocessing) ───────────────────
# Each class hard-codes its region + scenario so it can be pickled by name
# and re-instantiated in worker processes.

class UniformEquatorSampler(BudgetRegionalSampler):
    def __init__(self, seed=None, P_total=DEFAULT_P_TOTAL):
        super().__init__(*_EQ_T, UNIFORM[0], P_total, seed)

class UniformPoleSampler(BudgetRegionalSampler):
    def __init__(self, seed=None, P_total=DEFAULT_P_TOTAL):
        super().__init__(*_POLE_T, UNIFORM[1], P_total, seed)

class Soderlund2014EquatorSampler(BudgetRegionalSampler):
    def __init__(self, seed=None, P_total=DEFAULT_P_TOTAL):
        super().__init__(*_EQ_T, SODERLUND_2014[0], P_total, seed)

class Soderlund2014PoleSampler(BudgetRegionalSampler):
    def __init__(self, seed=None, P_total=DEFAULT_P_TOTAL):
        super().__init__(*_POLE_T, SODERLUND_2014[1], P_total, seed)

class Lemasquerier2023EquatorSampler(BudgetRegionalSampler):
    def __init__(self, seed=None, P_total=DEFAULT_P_TOTAL):
        super().__init__(*_EQ_T, LEMASQUERIER_2023_CONSERVATIVE[0], P_total, seed)

class Lemasquerier2023PoleSampler(BudgetRegionalSampler):
    def __init__(self, seed=None, P_total=DEFAULT_P_TOTAL):
        super().__init__(*_POLE_T, LEMASQUERIER_2023_CONSERVATIVE[1], P_total, seed)
