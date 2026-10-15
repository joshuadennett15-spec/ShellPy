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
| 4. Convection from the sampled rheology | 293 | 40.5 | 22.3-72.1 | 76% | 3637 |
| 5. Grid-converged lid base | 275 | 38.9 | 22.1-75.6 | 74% | 5008 |
| 6. Convergence by net drift | 280 | 39.1 | 22.0-76.1 | 75% | 5000 |
| 7. Both branches: thin start | 280 | 32.8 | 21.0-60.3 | 33% | 3900 |
| 7. Both branches: thick start | 280 | 39.1 | 22.0-76.1 | 75% | 5000 |
| 8. Labelled outcomes: thin start | 281 | 32.8 | 21.0-60.7 | 33% | 4000 |
| 8. Labelled outcomes: thick start | 281 | 39.3 | 22.1-76.1 | 75% | 5000 |
| 9. Unclipped viscosity: thin / thick | 281 | 32.8 / 39.3 | 21.0-60.7 / 22.1-76.1 | 33% / 75% | 4000 / 5000 |
| 11. DV2021 on the whole shell: thin / thick | 247 / 248 | 43.5 / 43.9 | 22.1-88.1 / 22.2-88.1 | 83% / 82% | 3800 / 4300 |

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
   selection is listed under known problems below.

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

4. **Convection from the sampled rheology.** v1 computed Ra and Nu from a fixed
   Frank-Kamenetskii viscosity (5e13 Pa s at the melting point, Q = 59.4 kJ/mol), so the
   sampled grain size, diffusivities and activation energies set the tidal heating but never
   the convection. Ra now uses the sampled diffusion-creep viscosity at the sublayer's mean
   temperature, and the viscosity contrast gamma is ln(eta(T_c) / eta(T_melt)) of the same
   law. The 0.75 factor on Nu, which is not in DV2021, is removed. Grain size now matters:
   the rank correlation of thickness with grain size goes from -0.03 to -0.35. Shells with
   coarse grains (median 1.3 mm) stay conductive and thin; the convecting ones have a median
   grain size of 0.46 mm. gamma is 3.1-3.3 across the convecting samples, below the range
   DV2021 calibrated their law on, so the Nu law itself still needs checking against the paper.

5. **Grid-converged lid base.** The convecting sublayer started at the first grid node warmer
   than T_c, so on the 31-node grid the lid base could only sit at a node, 1-3 km apart.
   Convecting shells got stuck at grid-dependent thicknesses: against a 241-node run of the
   same code, 1 sample in 10 was off by more than 2.6 km and the worst by 22 km, and some
   parameter sets showed two convecting equilibria that merged as the grid was refined. Now:
   - the lid-base depth extends the lid's temperature gradient to T_c (the sublayer side is
     Nu times flatter, so interpolating linearly across the cell misplaced it by up to half
     a cell);
   - the face the lid base crosses gets the lid and sublayer conductivities in series,
     weighted by where the lid base is, so convection starts at D_cond, not at the next node;
   - backward Euler replaces Crank-Nicolson. At dt = 1e12 s Crank-Nicolson leaves
     grid-scale oscillations undamped, and these grew once conductivity followed the lid
     base smoothly. The equilibrium does not depend on the time scheme;
   - 121 nodes instead of 31, at the same run time (Python overhead dominates).
   Against 241 nodes (60 samples), 9 in 10 samples are now within 0.03 km and the worst
   within 0.6 km. 20 samples now grow through the ocean (5 before): the grid had been holding
   some convecting shells too thin. 5 samples reach equilibrium but are flagged as not
   converged; see fix 6.

6. **Convergence by net drift.** At equilibrium H flickers by a few millimetres per step
   (|dH/dt| up to 6e-15 m/s, changing sign) because the Picard iteration stops at 0.01 K, so
   the per-step test of fix 1 (|dH/dt| < 1e-15 m/s for 200 steps in a row) never passed for
   5 samples that had stopped moving. A column now counts as converged when H moves less than
   0.2 m over a block of 200 steps, the same drift bound. All 300 samples converge; samples
   that converged before move by at most 0.12 m.

7. **Both branches.** Every solve now takes `start='thin'` (5 km, a shell growing from thin)
   or `start='thick'` (the old warm start: the conductive estimate times 8, clipped to
   5-100 km), and the two are run as separate ensembles with the same inputs. 2D columns use
   the same rule, replacing a 2D-only start (x1.5, clipped to 80 km). Of the 280 samples that
   are ok from both starts, 163 reach the same equilibrium (within 0.2 km), and 117 (42%)
   have two: conductive from the thin start and convecting from the thick one, a median
   6.6 km thicker (16-84%: 0.3-25.5 km, at most 69 km). No sample convects from the thin start
   and conducts from the thick one. Every row so far used the thick start, so rows 1-6 and
   the thick row describe the convecting branch.

8. **Labelled outcomes.** Every sample now keeps its outputs and says what happened to it,
   so training data can be filtered or modelled without guessing why rows are missing.
   `status` is 'ok', 'frozen through' (the shell reached the bottom of the hydrosphere: no
   ocean is left), 'not converged', 'melted through' (thinned to the 0.5 km floor) or the
   error raised. A column now stops as soon as it freezes through instead of growing past the
   seafloor, and the thick start is capped at 95% of the hydrosphere. v1 replaced the
   thickness of failed 2D columns by interpolating their neighbours, and called a 2D sample
   'ok' if half its columns were; now 2D keeps every column's real thickness with its own
   `column_status`, and the sample is 'ok' only if all of them are. The 19 'frozen through'
   samples (6%, the same from both starts) were the 'rejected: thickness' rows. One sample
   sits at 99.7% of its hydrosphere: the old 99% cut rejected it, now it is ok.

9. **Unclipped viscosity.** v1 clipped the diffusion-creep viscosity to 1e12-1e25 Pa s without
   recording it. The lower bound is reached at the melting point by about 1% of prior draws
   (the finest grains, down to 6e10 Pa s), where it changed Ra, Nu and tidal heating; the upper
   bound is reached near the surface by almost every draw, where tidal heating is negligible
   either way. The viscosity is now the creep law's own value. No sample of the 300 moves by
   more than 0.14 km, but the prior's fine-grain tail is no longer silently altered.

10. **One prior for 1D and 2D.** Howell (2021, Table 2) gives the surface temperature
    (104 +- 7 K) and tidal strain as global means, so a 2D sample is now the 1D sample with
    the same index, spread over latitude so that the area-weighted means are the 1D values:
    mean surface temperature = T_surf (sets the equatorial temperature, given the polar one),
    mean squared strain = epsilon_0^2 with a pole/equator strain ratio of 2, and mean ocean heat
    flux = the 1D basal flux. v1's 2D drew its own strains (equator 6e-6, pole 1.2e-5, 0.15 dex)
    beside the 1D 1.2e-5 at 0.3 dex, used the 1D mean temperature as the equatorial one, and
    multiplied the silicate tidal flux by 1.2; all three are gone. The Soderlund et al. (2014)
    scenario uses q* = 0.40, from their statement that the ocean heat flux "varies by about 40%
    with latitude, with a clear maximum at the equator" (the thesis archive used 0.40; the code
    had 0.45). 1D results are unchanged. 2D, 40 samples, 7 latitudes, thick start: median
    thickness at the equator / pole goes from 54.8 / 50.9 to 39.8 / 38.8 km (uniform ocean
    flux) and from 45.9 / 58.3 to 36.4 / 43.8 km (Soderlund).
    Open: the pole/equator strain ratio of 2 is v1's, citing Tobie et al. (2003), not yet checked
    against the paper; and a 104 K global mean puts the equator near 116 K, warmer than the
    96 K equatorial mean of Ashkenazy (2019), whose global mean is about 90 K.

11. **DV2021 convection applied as published.** Checked against the full text (ESSOAr preprint
    doi 10.1002/essoar.10507210.1). v1 applied the DV2021 heat-flux law (Eq. 23) to the
    convecting sublayer only, at gamma ~ 3, and took the interior temperature (c1 = 1.43,
    c2 = -0.03) and the lid-base temperature (2.24 viscous temperature scales) from Deschamps &
    Sotin (2001). DV2021 calibrated their laws on the whole stagnant-lid shell (viscosity ratio
    1e4-1e8, i.e. gamma 9.2-18.4; Ra_eff 2e5-2e8) and apply them to Europa that way (their SI
    S3). Now, for the whole shell (surface to melting point, thickness H):
    - gamma = Delta T / Delta T_v with Delta T_v = (-d ln eta / dT)^-1 of the sampled creep law
      (Eq. 22), Ra_eff at the interior temperature Tm (Eqs. 10, S19);
    - Tm from Eq. 21 including the tidal heating made in the shell, capped at the melting point;
    - surface flux (Eq. 23), lid thickness (Eq. 26), lid-base temperature (Eq. 19);
    - their procedure for a negative bottom flux: the Ur > 1 constants, then zero flux;
    - convection when its surface flux beats conduction (Eq. 14), replacing Ra > 1000, which is
      not in DV2021;
    - a convecting shell takes the DV2021 bottom heat flux (Eq. 11) from the ocean; the solved
      temperature profile (lid by conduction, sublayer conductivity times Nu so it carries the
      DV2021 flux from the lid base to the melting point) sets the tidal heating.
    DV2021's single conductivity is the conductivity integral of k = 612/T (with porosity) over
    the shell's temperature range, so their conductive limit matches the conducting column.
    Energy still balances (surface loss / supply within 0.5%).
    Results: the two equilibria of fix 7 were an artefact of the sublayer closure. Of the 247
    samples ok from both starts, 245 reach the same thickness and 2 differ by 0.8 and 1.1 km:
    near the convection onset a column can switch convection on and off, and these two settle
    into different bands. Such columns now count as converged when the band stops moving
    (`H_spread_km`, at most 0.17 km; 10 samples), so 1 sample is unconverged, down from 10-16.
    At equilibrium gamma is 13-19 and Ur < 1 in every sample. Labels: 35 samples have DV2021's
    interior at the melting point (`interior_melting`, their partial-melt regime) and 40 have
    gamma or Ra_eff outside the calibrated range (`outside_dv_calibration`, mostly Ra_eff up to
    8e9). 52 samples freeze the ocean through (19 before).

### Known problems

- DV2021 assume uniform internal heating; here it is the tidal heating of the solved profile,
  concentrated in the warm sublayer. Their Eq. 14 conductive limit therefore differs a little
  from the conducting column, which leaves the small onset hysteresis noted in fix 11.
- The pole/equator strain ratio of 2 and the 104 K global-mean surface temperature (fix 10).
