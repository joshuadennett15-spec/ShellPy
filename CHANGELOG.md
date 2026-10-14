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

### Known problems inherited from v1 (to be fixed)

1. Runs stop after 1500 steps or the first step with |dH/dt| < 1e-12 m/s, usually before
   equilibrium; v1 ensembles are biased by up to tens of km.
2. Convecting shells do not conserve energy: the Stefan condition uses plain k against a
   gradient flattened by Nu, so the surface loses 1.2-3x the heat supplied.
3. Convection onset is discontinuous and the equilibrium depends on the starting state.
4. Grain size and the sampled activation energy never reach Ra or Nu.
5. The sampled radiogenic flux q_rad is discarded and recomputed from H_rad.
6. The coordinate system switches to spherical only when the starting guess exceeds 30 km.
7. 1D and 2D priors differ (tidal strain, a 1.2x tidal flux factor in 2D).
