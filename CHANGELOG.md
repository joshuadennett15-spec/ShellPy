# Changelog

## 2.0.0 (in development)

### Restructure (results unchanged)

- One installable package, `src/shellpy`, replaces `Europa1D/` and `Europa2D/`. It reproduces
  v1.0.1 to within 1e-13 relative on 400 1D samples and on 12 2D samples, and runs 1.9x faster.
- Physics settings are fixed in code instead of read from a hand-edited `config.json`.
- Only the production physics is kept: Andrade tidal heating, diffusion creep, the
  Deschamps & Vilella (2021) Nusselt law, k = 612/T. Alternatives remain at tag v1.0.1.
- Analysis and figure scripts, alternative samplers, the Sobol workflow, and the committed
  results and figures were removed; all remain at tag v1.0.1.
- A sample that raises an error is recorded with the reason instead of silently dropped.
- 2D: latitude columns are independent (lateral heat flow removed; it changed the
  equilibrium thickness by at most 72 m in v1). 2D now uses the same Nusselt law as 1D;
  the v1 thesis 2D runs used the Green et al. (2021) law.

### Science fixes

Each fix is measured on the same 300-sample 1D ensemble (seed 10042).

| Change | ok | Median H (km) | 16-84% (km) | Convecting | Median steps |
|---|---|---|---|---|---|
| v1.0.1 behaviour | 300 | 32.5 | 17.5-54.1 | 80% | 1500 |
| 1. Run to equilibrium | 298 | 20.6 | 17.3-23.9 | 72% | 3890 |
| 2. Energy-conserving base | 295 | 40.2 | 22.1-78.3 | 100% | 3471 |
| 3. Priors and geometry | 293 | 37.2 | 22.3-76.1 | 100% | 4058 |

1. **Run to equilibrium.** v1 stopped after 1500 steps, or at the first step with
   |dH/dt| < 1e-12 m/s; most samples were still thinning from the warm start. A column
   now counts as converged when |dH/dt| < 1e-15 m/s for 200 consecutive steps (under 0.2 m
   of drift), within 50,000 steps, and every result records `converged`. The v1 upper tail
   (median 32.5 km, 84th percentile 54 km) was an artefact of stopping early. 2D columns now
   converge independently instead of stepping until the slowest one stops.

2. **Energy-conserving base.** v1's Stefan condition multiplied plain k by a basal gradient
   that Nu had flattened, so a convecting shell lost 1.2-3x more heat at its surface than
   the ocean and tides supplied (2.85x in the test shell). The flux into the ice is now the
   flux carried through the bottom cell face, Nu-enhanced where convecting, minus the tidal
   heat of the bottom half cell; surface loss matches supply to 1e-4. Correct energy makes
   convecting shells thicker, and exposes two equilibria for many parameter sets: a thinner
   conductive shell and a thicker convecting one (Mitri & Showman 2005). Which one a run
   reaches depends on where it starts; the v1 warm start begins thick (up to 100 km), so
   every sample above lands on the convecting branch and 4 grow through the ocean. Branch
   selection is open problem 4 below.

3. **Priors and geometry.** The basal heat flux is now the sum of the two sampled fluxes,
   q_radiogenic ~ N(7, 1) truncated to 4-12 mW/m^2 and q_tidal_silicate ~ LogUniform(2, 20)
   mW/m^2. v1 drew q_rad and discarded it, using H_rad x rock mass / area instead, so the
   hydrosphere depth no longer changes the basal flux. Every prior is a plain draw from its
   stated distribution: v1's discarded draws, clip-then-redraw steps and the always-zero salt
   terms are gone. The column is always a spherical shell; v1 used Cartesian geometry when
   the starting guess was under 30 km. The solver matches the exact steady state of a
   conducting spherical shell (k = 612/T) to 1e-4. The draw order changed, so seed 10042 now
   gives different parameter sets. The 3 km drop in the median from row 2 is about 1.4
   standard errors of the difference (bootstrap SE 1.5 km per median), so it cannot be told
   apart from resampling. 2D: the 1.2x tidal flux factor is now an explicit
   `solve_2d(q_tidal_scale=1.2)` argument instead of being baked into the sample.

### Known problems inherited from v1 (to be fixed)

4. Convection onset is discontinuous and the equilibrium depends on the starting state.
5. Grain size and the sampled activation energy never reach Ra or Nu.
6. 1D and 2D priors differ (tidal strain, a 1.2x tidal flux factor in 2D).
