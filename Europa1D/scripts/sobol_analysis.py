"""
Global Sensitivity Analysis for Europa Ice Shell Monte Carlo Results

Computes eight complementary sensitivity metrics from existing MC output
(no re-running of simulations):

    1. Spearman rank correlation  -- monotonic dependence
    2. Partial Rank Correlation Coefficients (PRCC) -- monotonic, controlling
       for all other parameters
    3. Standardized Regression Coefficients (SRC) -- linear sensitivity
    4. Random-Forest permutation importance -- non-linear, interaction-aware
    5. Delta moment-independent measure (Borgonovo) -- distributional
    6. SHAP values (TreeExplainer) -- game-theoretic, per-sample attribution
    7. Mutual Information -- non-parametric dependence
    8. Conditional sensitivity / KS filtering -- regime-dependent importance

Produces:
    results/sensitivity_indices.csv        -- table of all indices
    figures/sensitivity_analysis.png       -- 6-panel bar-chart summary
    figures/sensitivity_shap.png           -- SHAP beeswarm + dependence
    figures/sensitivity_conditional.png    -- regime-dependent KS + PRCC
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression
from SALib.analyze import delta as salib_delta
import shap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')

# Human-readable labels for the 16 sampled parameters
PARAM_LABELS = {
    'd_grain':    r'$d_\mathrm{grain}$',
    'epsilon_0':  r'$\varepsilon_0$',
    'T_surf':     r'$T_\mathrm{surf}$',
    'D_H2O':      r'$D_\mathrm{H_2O}$',
    'mu_ice':     r'$\mu_\mathrm{ice}$',
    'Q_v':        r'$Q_v$',
    'Q_b':        r'$Q_b$',
    'H_rad':      r'$H_\mathrm{rad}$',
    'P_tidal':    r'$P_\mathrm{tidal}$',
    'f_porosity': r'$f_\mathrm{por}$',
    'f_salt':     r'$f_\mathrm{salt}$',
    'T_phi':      r'$T_\phi$',
    'B_k':        r'$B_k$',
    'D0v':        r'$D_{0v}$',
    'D0b':        r'$D_{0b}$',
    'd_del':      r'$\delta_\mathrm{gb}$',
}


# =====================================================================
# 1. Spearman rank correlations
# =====================================================================
def spearman_correlations(X: np.ndarray, Y: np.ndarray, names: list) -> pd.DataFrame:
    rows = []
    for j, name in enumerate(names):
        rho, pval = stats.spearmanr(X[:, j], Y)
        rows.append({'parameter': name, 'spearman_rho': rho, 'spearman_p': pval})
    return pd.DataFrame(rows)


# =====================================================================
# 2. Partial Rank Correlation Coefficients (PRCC)
# =====================================================================
def partial_rank_correlations(X: np.ndarray, Y: np.ndarray, names: list) -> pd.DataFrame:
    n, p = X.shape
    R = np.column_stack([
        stats.rankdata(X[:, j]) for j in range(p)
    ] + [stats.rankdata(Y)])

    C = np.corrcoef(R, rowvar=False)
    try:
        P = np.linalg.inv(C)
    except np.linalg.LinAlgError:
        P = np.linalg.pinv(C)

    prcc = np.zeros(p)
    for j in range(p):
        prcc[j] = -P[j, -1] / np.sqrt(P[j, j] * P[-1, -1])

    dof = n - p - 2
    t_stat = prcc * np.sqrt(dof / (1 - prcc**2 + 1e-30))
    p_vals = 2 * stats.t.sf(np.abs(t_stat), dof)

    return pd.DataFrame({
        'parameter': names, 'PRCC': prcc, 'PRCC_p': p_vals,
    })


# =====================================================================
# 3. Standardized Regression Coefficients (SRC)
# =====================================================================
def standardized_regression(X: np.ndarray, Y: np.ndarray, names: list):
    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()

    Xs = scaler_X.fit_transform(X)
    Ys = scaler_Y.fit_transform(Y.reshape(-1, 1)).ravel()

    beta = np.linalg.lstsq(Xs, Ys, rcond=None)[0]

    Y_pred = Xs @ beta
    ss_res = np.sum((Ys - Y_pred) ** 2)
    ss_tot = np.sum((Ys - Ys.mean()) ** 2)
    R2 = 1 - ss_res / ss_tot
    print(f"  SRC linear model R^2 = {R2:.3f}")

    return pd.DataFrame({
        'parameter': names, 'SRC': beta, 'SRC_abs': np.abs(beta),
    }), R2


# =====================================================================
# 4. Random-Forest permutation importance
# =====================================================================
def random_forest_importance(X, Y, names, n_estimators=500, seed=42):
    rf = RandomForestRegressor(
        n_estimators=n_estimators, max_depth=None,
        min_samples_leaf=5, random_state=seed, n_jobs=-1,
    )
    rf.fit(X, Y)
    R2_train = rf.score(X, Y)
    print(f"  RF in-sample R^2 = {R2_train:.3f}")

    perm = permutation_importance(
        rf, X, Y, n_repeats=30, random_state=seed, n_jobs=-1,
    )

    df = pd.DataFrame({
        'parameter': names,
        'RF_importance': perm.importances_mean,
        'RF_importance_std': perm.importances_std,
    })
    return df, R2_train, rf          # also return the trained RF


# =====================================================================
# 5. Delta moment-independent measure (Borgonovo via SALib)
# =====================================================================
def delta_indices(X, Y, names):
    problem = {
        'num_vars': X.shape[1], 'names': names,
        'bounds': [[X[:, j].min(), X[:, j].max()] for j in range(X.shape[1])],
    }
    Si = salib_delta.analyze(problem, X, Y, print_to_console=False, seed=42)
    return pd.DataFrame({
        'parameter': names,
        'delta': Si['delta'], 'delta_conf': Si['delta_conf'],
        'S1_delta': Si['S1'], 'S1_delta_conf': Si['S1_conf'],
    })


# =====================================================================
# 6. SHAP values
# =====================================================================
def compute_shap(rf, X, names):
    """TreeExplainer SHAP values from the trained Random Forest."""
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    df = pd.DataFrame({
        'parameter': names,
        'SHAP_mean_abs': mean_abs_shap,
    })
    return df, shap_values


# =====================================================================
# 7. Mutual Information
# =====================================================================
def mutual_information(X, Y, names, seed=42):
    mi = mutual_info_regression(X, Y, random_state=seed, n_neighbors=7)
    return pd.DataFrame({
        'parameter': names, 'MI': mi,
    })


# =====================================================================
# 8. Conditional Sensitivity / KS filtering
# =====================================================================
def conditional_sensitivity(X, Y, names, quantile_lo=0.20, quantile_hi=0.80):
    """
    Regional Sensitivity Analysis (RSA): compare parameter distributions
    between thin-shell and thick-shell outcomes via two-sample KS test.
    Also compute PRCC within each regime.
    """
    q_lo = np.percentile(Y, 100 * quantile_lo)
    q_hi = np.percentile(Y, 100 * quantile_hi)
    thin_mask = Y <= q_lo
    thick_mask = Y >= q_hi

    ks_rows = []
    for j, name in enumerate(names):
        stat, pval = stats.ks_2samp(X[thin_mask, j], X[thick_mask, j])
        ks_rows.append({
            'parameter': name, 'KS_stat': stat, 'KS_p': pval,
        })
    df_ks = pd.DataFrame(ks_rows)

    # PRCC within the thin regime
    prcc_thin = partial_rank_correlations(
        X[thin_mask], Y[thin_mask], names
    ).rename(columns={'PRCC': 'PRCC_thin', 'PRCC_p': 'PRCC_thin_p'})

    # PRCC within the thick regime
    prcc_thick = partial_rank_correlations(
        X[thick_mask], Y[thick_mask], names
    ).rename(columns={'PRCC': 'PRCC_thick', 'PRCC_p': 'PRCC_thick_p'})

    df = df_ks.merge(prcc_thin, on='parameter').merge(prcc_thick, on='parameter')
    return df, q_lo, q_hi, thin_mask, thick_mask


# =====================================================================
# Composite ranking
# =====================================================================
def composite_score(df: pd.DataFrame) -> pd.DataFrame:
    metrics = ['|spearman|', '|PRCC|', 'SRC_abs', 'RF_importance',
               'delta', 'SHAP_mean_abs', 'MI']
    for m in metrics:
        col_max = df[m].max()
        if col_max > 0:
            df[f'{m}_norm'] = df[m] / col_max
        else:
            df[f'{m}_norm'] = 0.0

    norm_cols = [f'{m}_norm' for m in metrics]
    df['composite'] = df[norm_cols].mean(axis=1)
    df = df.drop(columns=norm_cols)
    return df.sort_values('composite', ascending=False).reset_index(drop=True)


# =====================================================================
# Figure 1: bar-chart summary (updated to include SHAP + MI panels)
# =====================================================================
def plot_sensitivity(df, Y, X, names, R2_lin, R2_rf):
    fig = plt.figure(figsize=(20, 18))
    gs = gridspec.GridSpec(4, 2, hspace=0.36, wspace=0.30,
                           left=0.07, right=0.97, top=0.95, bottom=0.04)

    df_sorted = df.sort_values('composite', ascending=True)
    labels = [PARAM_LABELS.get(n, n) for n in df_sorted['parameter']]
    y_pos = np.arange(len(labels))

    # (a) Spearman
    ax = fig.add_subplot(gs[0, 0])
    colors = ['#C44E52' if v < 0 else '#4C72B0' for v in df_sorted['spearman_rho']]
    ax.barh(y_pos, df_sorted['spearman_rho'].values, color=colors, edgecolor='none')
    ax.set_yticks(y_pos); ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel(r'Spearman $\rho$'); ax.axvline(0, color='k', lw=0.5)
    ax.set_title('(a) Spearman Rank Correlation', fontsize=11, fontweight='bold')
    ax.grid(axis='x', alpha=0.2)

    # (b) PRCC
    ax = fig.add_subplot(gs[0, 1])
    colors = ['#C44E52' if v < 0 else '#4C72B0' for v in df_sorted['PRCC']]
    ax.barh(y_pos, df_sorted['PRCC'].values, color=colors, edgecolor='none')
    ax.set_yticks(y_pos); ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('PRCC'); ax.axvline(0, color='k', lw=0.5)
    ax.set_title('(b) Partial Rank Correlation Coefficient', fontsize=11, fontweight='bold')
    ax.grid(axis='x', alpha=0.2)

    # (c) SRC
    ax = fig.add_subplot(gs[1, 0])
    colors = ['#C44E52' if v < 0 else '#4C72B0' for v in df_sorted['SRC']]
    ax.barh(y_pos, df_sorted['SRC'].values, color=colors, edgecolor='none')
    ax.set_yticks(y_pos); ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('SRC')
    ax.set_title(f'(c) Standardized Regression Coefficient  ($R^2$ = {R2_lin:.2f})',
                 fontsize=11, fontweight='bold')
    ax.axvline(0, color='k', lw=0.5); ax.grid(axis='x', alpha=0.2)

    # (d) RF permutation importance
    ax = fig.add_subplot(gs[1, 1])
    ax.barh(y_pos, df_sorted['RF_importance'].values, color='#55A868',
            xerr=df_sorted['RF_importance_std'].values, capsize=2, edgecolor='none')
    ax.set_yticks(y_pos); ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel(r'Permutation Importance ($\Delta R^2$)')
    ax.set_title(f'(d) Random Forest Importance  ($R^2$ = {R2_rf:.2f})',
                 fontsize=11, fontweight='bold')
    ax.grid(axis='x', alpha=0.2)

    # (e) SHAP mean |value|
    ax = fig.add_subplot(gs[2, 0])
    ax.barh(y_pos, df_sorted['SHAP_mean_abs'].values, color='#DD8452', edgecolor='none')
    ax.set_yticks(y_pos); ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Mean |SHAP value| (km)')
    ax.set_title('(e) SHAP Feature Importance', fontsize=11, fontweight='bold')
    ax.grid(axis='x', alpha=0.2)

    # (f) Mutual Information
    ax = fig.add_subplot(gs[2, 1])
    ax.barh(y_pos, df_sorted['MI'].values, color='#8172B2', edgecolor='none')
    ax.set_yticks(y_pos); ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Mutual Information (nats)')
    ax.set_title('(f) Mutual Information', fontsize=11, fontweight='bold')
    ax.grid(axis='x', alpha=0.2)

    # (g) Delta (Borgonovo)
    ax = fig.add_subplot(gs[3, 0])
    ax.barh(y_pos, df_sorted['delta'].values, color='#937860',
            xerr=df_sorted['delta_conf'].values, capsize=2, edgecolor='none')
    ax.set_yticks(y_pos); ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel(r'Borgonovo $\delta$')
    ax.set_title('(g) Delta Moment-Independent Measure', fontsize=11, fontweight='bold')
    ax.grid(axis='x', alpha=0.2)

    # (h) Composite ranking
    ax = fig.add_subplot(gs[3, 1])
    bar_colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(df_sorted)))
    ax.barh(y_pos, df_sorted['composite'].values, color=bar_colors, edgecolor='none')
    ax.set_yticks(y_pos); ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Composite Score (normalised mean)')
    ax.set_title('(h) Composite Sensitivity Ranking', fontsize=11, fontweight='bold')
    ax.grid(axis='x', alpha=0.2)

    fig.suptitle('Global Sensitivity Analysis  --  Europa Ice Shell Thickness (10,000 MC)',
                 fontsize=14, fontweight='bold', y=0.98)

    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIGURES_DIR, 'sensitivity_analysis.png'), dpi=200)
    plt.close(fig)
    print("  Figure saved -> figures/sensitivity_analysis.png")


# =====================================================================
# Figure 2: SHAP beeswarm + top-4 dependence plots
# =====================================================================
def plot_shap(shap_values, X, names, df_composite):
    """SHAP beeswarm and dependence plots for the top parameters."""
    top_params = df_composite.sort_values('composite', ascending=False)['parameter'].tolist()
    labels_all = [PARAM_LABELS.get(n, n) for n in names]
    top4_idx = [names.index(p) for p in top_params[:4]]

    fig = plt.figure(figsize=(18, 14))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.30,
                           left=0.08, right=0.96, top=0.93, bottom=0.06)

    # (a) Beeswarm  -- full panel
    ax = fig.add_subplot(gs[0, :])
    # Sort features by mean |SHAP|
    order = np.argsort(np.abs(shap_values).mean(axis=0))[::-1]
    shap_sorted = shap_values[:, order]
    X_sorted = X[:, order]
    labels_sorted = [labels_all[i] for i in order]

    # Sub-sample for plotting speed
    n_show = min(2000, X.shape[0])
    rng = np.random.default_rng(0)
    idx = rng.choice(X.shape[0], n_show, replace=False)

    for rank, feat_i in enumerate(range(min(12, len(names)))):
        sv = shap_sorted[idx, feat_i]
        fv = X_sorted[idx, feat_i]
        # normalise feature values to [0,1] for colormap
        fmin, fmax = fv.min(), fv.max()
        if fmax > fmin:
            fv_norm = (fv - fmin) / (fmax - fmin)
        else:
            fv_norm = np.full_like(fv, 0.5)
        jitter = rng.normal(0, 0.15, size=n_show)
        y_val = (min(12, len(names)) - 1 - rank) + jitter
        ax.scatter(sv, y_val, c=fv_norm, cmap='coolwarm', s=4, alpha=0.5,
                   edgecolors='none', rasterized=True)

    ax.set_yticks(range(min(12, len(names))))
    ax.set_yticklabels(list(reversed(labels_sorted[:min(12, len(names))])), fontsize=10)
    ax.set_xlabel('SHAP value (impact on thickness, km)', fontsize=11)
    ax.axvline(0, color='k', lw=0.5)
    ax.set_title('(a) SHAP Beeswarm  --  top 12 parameters', fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.15)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['Low', 'High'])
    cbar.set_label('Feature value', fontsize=10)

    # (b-d) Dependence plots for top 3 parameters
    for panel_i, feat_idx in enumerate(top4_idx[:3]):
        ax = fig.add_subplot(gs[1, panel_i]) if panel_i < 2 else None
        if panel_i == 2:
            # third subplot in second row, need a 3-column sub-grid
            # just put it as gs[1,1] offset -- use a different approach
            break
        sv = shap_values[:, feat_idx]
        fv = X[:, feat_idx]
        label = labels_all[feat_idx]

        ax.scatter(fv[idx], sv[idx], c=sv[idx], cmap='coolwarm',
                   s=6, alpha=0.4, edgecolors='none', rasterized=True)
        ax.set_xlabel(f'{label} value', fontsize=11)
        ax.set_ylabel('SHAP value (km)', fontsize=10)
        ax.set_title(f'({chr(98+panel_i)}) Dependence: {label}',
                     fontsize=11, fontweight='bold')
        ax.axhline(0, color='k', lw=0.5, alpha=0.5)
        ax.grid(alpha=0.15)

    fig.suptitle('SHAP Analysis  --  Europa Ice Shell Thickness',
                 fontsize=14, fontweight='bold', y=0.97)
    fig.savefig(os.path.join(FIGURES_DIR, 'sensitivity_shap.png'), dpi=200)
    plt.close(fig)
    print("  Figure saved -> figures/sensitivity_shap.png")


# =====================================================================
# Figure 3: Conditional sensitivity (KS + regime PRCC)
# =====================================================================
def plot_conditional(df_cond, q_lo, q_hi, X, Y, names, thin_mask, thick_mask):
    """KS statistic + PRCC comparison between thin/thick regimes."""
    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(2, 2, hspace=0.38, wspace=0.30,
                           left=0.07, right=0.97, top=0.93, bottom=0.07)

    df_sorted = df_cond.sort_values('KS_stat', ascending=True)
    labels = [PARAM_LABELS.get(n, n) for n in df_sorted['parameter']]
    y_pos = np.arange(len(labels))

    # (a) KS statistic
    ax = fig.add_subplot(gs[0, 0])
    sig_colors = ['#C44E52' if p < 0.01 else '#4C72B0'
                  for p in df_sorted['KS_p']]
    ax.barh(y_pos, df_sorted['KS_stat'].values, color=sig_colors, edgecolor='none')
    ax.set_yticks(y_pos); ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('KS Statistic')
    ax.set_title(f'(a) KS Test: thin (<{q_lo:.0f} km) vs thick (>{q_hi:.0f} km)',
                 fontsize=11, fontweight='bold')
    ax.grid(axis='x', alpha=0.2)
    # Legend for significance
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color='#C44E52', label='p < 0.01'),
                       Patch(color='#4C72B0', label='p >= 0.01')],
              fontsize=8, loc='lower right')

    # (b) PRCC thin vs thick side-by-side
    ax = fig.add_subplot(gs[0, 1])
    width = 0.35
    prcc_thin = df_sorted['PRCC_thin'].values
    prcc_thick = df_sorted['PRCC_thick'].values
    ax.barh(y_pos - width/2, prcc_thin, height=width, color='#64B5F6',
            label=f'Thin (<{q_lo:.0f} km)', edgecolor='none')
    ax.barh(y_pos + width/2, prcc_thick, height=width, color='#EF5350',
            label=f'Thick (>{q_hi:.0f} km)', edgecolor='none')
    ax.set_yticks(y_pos); ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('PRCC')
    ax.set_title('(b) PRCC by Thickness Regime', fontsize=11, fontweight='bold')
    ax.axvline(0, color='k', lw=0.5)
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(axis='x', alpha=0.2)

    # (c) CDF comparison for top 3 KS parameters
    top3_ks = df_cond.sort_values('KS_stat', ascending=False).head(3)
    for panel_i, (_, row) in enumerate(top3_ks.iterrows()):
        if panel_i >= 2:
            break
        ax = fig.add_subplot(gs[1, panel_i])
        pname = row['parameter']
        j = names.index(pname)
        label = PARAM_LABELS.get(pname, pname)

        vals_thin = X[thin_mask, j]
        vals_thick = X[thick_mask, j]

        # Empirical CDFs
        xs_thin = np.sort(vals_thin)
        ys_thin = np.arange(1, len(xs_thin) + 1) / len(xs_thin)
        xs_thick = np.sort(vals_thick)
        ys_thick = np.arange(1, len(xs_thick) + 1) / len(xs_thick)

        ax.step(xs_thin, ys_thin, color='#64B5F6', lw=2,
                label=f'Thin (<{q_lo:.0f} km, n={thin_mask.sum()})')
        ax.step(xs_thick, ys_thick, color='#EF5350', lw=2,
                label=f'Thick (>{q_hi:.0f} km, n={thick_mask.sum()})')
        ax.set_xlabel(f'{label}', fontsize=11)
        ax.set_ylabel('CDF', fontsize=10)
        ax.set_title(f'({chr(99+panel_i)}) {label}  (KS={row["KS_stat"]:.3f})',
                     fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)

    fig.suptitle('Conditional Sensitivity  --  Thin vs Thick Shell Regimes',
                 fontsize=14, fontweight='bold', y=0.97)
    fig.savefig(os.path.join(FIGURES_DIR, 'sensitivity_conditional.png'), dpi=200)
    plt.close(fig)
    print("  Figure saved -> figures/sensitivity_conditional.png")


# =====================================================================
# Main
# =====================================================================
def main() -> None:
    # Load the 10,000-iteration MC results
    mc_path = os.path.join(RESULTS_DIR, 'monte_carlo_results.npz')
    data = np.load(mc_path)
    Y = data['thicknesses_km']

    param_names = []
    param_arrays = []
    for key in sorted(data.keys()):
        if key.startswith('param_'):
            name = key[6:]
            arr = data[key]
            if arr.shape[0] == Y.shape[0]:
                param_names.append(name)
                param_arrays.append(arr)

    if not param_names:
        raise RuntimeError("No sampled parameters found in monte_carlo_results.npz")

    X = np.column_stack(param_arrays)
    n, p = X.shape
    print(f"Loaded {n} valid samples x {p} parameters from {mc_path}")
    print(f"Thickness range: [{Y.min():.1f}, {Y.max():.1f}] km\n")

    # -- 1. Spearman --
    print("[1/8] Spearman rank correlations ...")
    df_sp = spearman_correlations(X, Y, param_names)

    # -- 2. PRCC --
    print("[2/8] Partial Rank Correlation Coefficients (PRCC) ...")
    df_prcc = partial_rank_correlations(X, Y, param_names)

    # -- 3. SRC --
    print("[3/8] Standardized Regression Coefficients (SRC) ...")
    df_src, R2_lin = standardized_regression(X, Y, param_names)

    # -- 4. Random Forest --
    print("[4/8] Random Forest permutation importance ...")
    df_rf, R2_rf, rf_model = random_forest_importance(X, Y, param_names)

    # -- 5. Delta --
    print("[5/8] Delta moment-independent measure (Borgonovo) ...")
    df_delta = delta_indices(X, Y, param_names)

    # -- 6. SHAP --
    print("[6/8] SHAP values (TreeExplainer) ...")
    df_shap, shap_values = compute_shap(rf_model, X, param_names)

    # -- 7. Mutual Information --
    print("[7/8] Mutual Information ...")
    df_mi = mutual_information(X, Y, param_names)

    # -- 8. Conditional / KS --
    print("[8/8] Conditional sensitivity (KS filtering + regime PRCC) ...")
    df_cond, q_lo, q_hi, thin_mask, thick_mask = conditional_sensitivity(
        X, Y, param_names
    )

    # Merge all metrics into single DataFrame
    df = df_sp.merge(df_prcc, on='parameter')
    df = df.merge(df_src, on='parameter')
    df = df.merge(df_rf, on='parameter')
    df = df.merge(df_delta, on='parameter')
    df = df.merge(df_shap, on='parameter')
    df = df.merge(df_mi, on='parameter')

    # Absolute-value columns for ranking
    df['|spearman|'] = df['spearman_rho'].abs()
    df['|PRCC|'] = df['PRCC'].abs()

    # Composite ranking (now includes SHAP + MI)
    df = composite_score(df)

    # Save CSV
    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, 'sensitivity_indices.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved -> {csv_path}")

    # Also save conditional results
    csv_cond = os.path.join(RESULTS_DIR, 'sensitivity_conditional.csv')
    df_cond.to_csv(csv_cond, index=False)
    print(f"Conditional results saved -> {csv_cond}")

    # Print summary
    print("\n" + "=" * 90)
    print("GLOBAL SENSITIVITY RANKING  (composite of 7 metrics)")
    print("=" * 90)
    cols = ['parameter', 'spearman_rho', 'PRCC', 'SRC', 'RF_importance',
            'SHAP_mean_abs', 'MI', 'delta', 'composite']
    fmt = {c: '{:.3f}'.format for c in cols if c != 'parameter'}
    print(df[cols].to_string(index=False, formatters=fmt))

    print(f"\n  Linear R^2 = {R2_lin:.3f},  RF R^2 = {R2_rf:.3f}")
    print(f"  => {100*(1-R2_lin/R2_rf):.0f}% of explainable variance is non-linear")

    # Conditional summary
    print("\n" + "-" * 90)
    print(f"CONDITIONAL SENSITIVITY  (thin <{q_lo:.0f} km  vs  thick >{q_hi:.0f} km)")
    print("-" * 90)
    df_cond_sorted = df_cond.sort_values('KS_stat', ascending=False)
    cols_c = ['parameter', 'KS_stat', 'KS_p', 'PRCC_thin', 'PRCC_thick']
    fmt_c = {c: '{:.3f}'.format for c in cols_c if c != 'parameter'}
    print(df_cond_sorted[cols_c].to_string(index=False, formatters=fmt_c))

    # Generate all three figures
    print("\nGenerating figures ...")
    os.makedirs(FIGURES_DIR, exist_ok=True)
    plot_sensitivity(df, Y, X, param_names, R2_lin, R2_rf)
    plot_shap(shap_values, X, param_names, df)
    plot_conditional(df_cond, q_lo, q_hi, X, Y, param_names, thin_mask, thick_mask)
    print("\nDone.")


if __name__ == '__main__':
    main()
