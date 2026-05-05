import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from runtime_support import configure_numeric_runtime, resolve_worker_count

configure_numeric_runtime()

import numpy as np
import multiprocessing as mp

from Monte_Carlo import MonteCarloRunner, SolverConfig
from regional_samplers_500 import (
    Run1EquatorSampler, Run1PoleSampler,
    Run2EquatorSampler, Run2PoleSampler,
    Run3EquatorSampler, Run3PoleSampler
)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_structure_comparison(equator, pole, save_path, title):
    """Plot conductive vs convective thickness using 2D KDE contours for equator vs pole."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Build combined dataframe for seaborn
    df_equator = pd.DataFrame({
        'D_cond': equator.D_cond_km,
        'D_conv': equator.D_conv_km,
        'Region': 'Equator'
    })
    df_pole = pd.DataFrame({
        'D_cond': pole.D_cond_km,
        'D_conv': pole.D_conv_km,
        'Region': 'Pole'
    })
    df = pd.concat([df_equator, df_pole], ignore_index=True)
    
    # Color palette
    palette = {'Equator': '#2E86AB', 'Pole': '#A23B72'}
    
    # 2D KDE contour plot - filled contours showing density
    sns.kdeplot(
        data=df,
        x='D_cond',
        y='D_conv',
        hue='Region',
        fill=True,
        alpha=0.5,
        levels=8,
        palette=palette,
        ax=ax,
        common_norm=False,
    )
    
    # Add contour lines for clarity
    sns.kdeplot(
        data=df,
        x='D_cond',
        y='D_conv',
        hue='Region',
        fill=False,
        levels=5,
        palette=palette,
        ax=ax,
        linewidths=1.5,
        common_norm=False,
    )
    
    # Calculate means for annotation
    eq_Dcond_mean = np.mean(equator.D_cond_km)
    eq_Dconv_mean = np.mean(equator.D_conv_km)
    pole_Dcond_mean = np.mean(pole.D_cond_km)
    pole_Dconv_mean = np.mean(pole.D_conv_km)
    
    # Mark mean positions with distinct markers
    ax.plot(eq_Dcond_mean, eq_Dconv_mean, 'o', color='#2E86AB', 
            markersize=14, markeredgecolor='white', markeredgewidth=2.5,
            label=f'Equator mean ({eq_Dcond_mean:.1f}, {eq_Dconv_mean:.1f}) km', zorder=10)
    ax.plot(pole_Dcond_mean, pole_Dconv_mean, 's', color='#A23B72', 
            markersize=14, markeredgecolor='white', markeredgewidth=2.5,
            label=f'Pole mean ({pole_Dcond_mean:.1f}, {pole_Dconv_mean:.1f}) km', zorder=10)
    
    # Reference lines for total thickness (D_cond + D_conv = H)
    max_x = max(equator.D_cond_km.max(), pole.D_cond_km.max()) * 1.05
    max_y = max(equator.D_conv_km.max(), pole.D_conv_km.max()) * 1.05
    
    for H_total in [10, 20, 30, 40, 50, 60, 70, 80]:
        x_line = np.array([0, min(H_total, max_x)])
        y_line = H_total - x_line
        valid = (y_line >= 0) & (y_line <= max_y)
        if np.any(valid):
            ax.plot(x_line[valid], y_line[valid], 'k--', alpha=0.25, lw=1)
            label_x = min(H_total * 0.85, max_x * 0.9)
            label_y = H_total - label_x
            if 0 < label_y < max_y * 0.9:
                ax.text(label_x, label_y, f'H={H_total}', fontsize=8, alpha=0.5, 
                        rotation=-45, ha='center', va='center')
    
    ax.set_xlabel("Conductive Lid Thickness, D_cond (km)", fontsize=12)
    ax.set_ylabel("Convective Layer Thickness, D_conv (km)", fontsize=12)
    ax.set_title(f"Structure Density: {title}", fontsize=14, fontweight='bold')
    ax.set_xlim(0, max_x)
    ax.set_ylim(0, max_y)
    ax.grid(True, alpha=0.3, linestyle=':')
    
    # Custom legend
    handles, labels = ax.get_legend_handles_labels()
    # Filter to only show the mean markers in legend
    ax.legend(loc='upper right', fontsize=10)
    
    # Add statistics annotation box
    eq_lid_frac = np.mean(equator.lid_fractions)
    pole_lid_frac = np.mean(pole.lid_fractions)
    eq_H_mean = np.mean(equator.thicknesses_km)
    pole_H_mean = np.mean(pole.thicknesses_km)
    
    stats_text = (
        f"EQUATOR (n={equator.n_valid}):\n"
        f"  Mean H: {eq_H_mean:.1f} km\n"
        f"  Lid fraction: {eq_lid_frac:.1%}\n"
        f"  D_cond: {eq_Dcond_mean:.1f} km\n"
        f"  D_conv: {eq_Dconv_mean:.1f} km\n\n"
        f"POLE (n={pole.n_valid}):\n"
        f"  Mean H: {pole_H_mean:.1f} km\n"
        f"  Lid fraction: {pole_lid_frac:.1%}\n"
        f"  D_cond: {pole_Dcond_mean:.1f} km\n"
        f"  D_conv: {pole_Dconv_mean:.1f} km"
    )
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='left',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))
    
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)

def plot_thickness_comparison(equator, pole, save_path, title):
    """Plot overlaid thickness distributions for equator vs pole."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Color scheme
    eq_color = "#2E86AB"   # Blue for equator (warmer region)
    pole_color = "#A23B72" # Magenta for pole (colder region)
    
    # Determine common bin range
    all_thickness = np.concatenate([equator.thicknesses_km, pole.thicknesses_km])
    bins = np.linspace(0, max(all_thickness) * 1.05, 40)
    
    # Plot histograms
    ax.hist(equator.thicknesses_km, bins=bins, density=True, alpha=0.4, 
            color=eq_color, label=f"Equator (n={equator.n_valid})")
    ax.hist(pole.thicknesses_km, bins=bins, density=True, alpha=0.4, 
            color=pole_color, label=f"Pole (n={pole.n_valid})")
    
    # Plot smoothed PDFs
    ax.plot(equator.bin_centers, equator.pdf_smoothed, color=eq_color, 
            lw=2.5, label=f"Equator PDF (CBE={equator.cbe_km:.1f} km)")
    ax.plot(pole.bin_centers, pole.pdf_smoothed, color=pole_color, 
            lw=2.5, label=f"Pole PDF (CBE={pole.cbe_km:.1f} km)")
    
    # Vertical lines for CBE
    ax.axvline(equator.cbe_km, color=eq_color, linestyle='--', lw=1.5, alpha=0.8)
    ax.axvline(pole.cbe_km, color=pole_color, linestyle='--', lw=1.5, alpha=0.8)
    
    # Labels and styling
    ax.set_xlabel("Ice Shell Thickness (km)", fontsize=12)
    ax.set_ylabel("Probability Density", fontsize=12)
    ax.set_title(f"Total Thickness: {title}", fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 80)
    ax.set_ylim(0, None)
    
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close(fig)

def run_scenarios(n_iterations=2500, n_workers=None, seed=42):
    n_workers = resolve_worker_count(n_workers)
    base_kwargs = dict(
        nx=31,
        initial_thickness=20e3,
        dt=1e12,
        total_time=5e14,
        eq_threshold=1e-12,
        max_steps=1500,
        use_convection=True,
        rannacher_steps=4,
        use_warm_start=True,
    )
    
    equator_config = SolverConfig(**base_kwargs, reject_subcritical=True)
    pole_config = SolverConfig(**base_kwargs, reject_subcritical=False)
    
    scenarios = [
        ("Run 1: Polar Heating", "run1_polar", Run1EquatorSampler, Run1PoleSampler),
        ("Run 2: Equatorial Heating", "run2_equatorial", Run2EquatorSampler, Run2PoleSampler),
        ("Run 3: Uniform Heating", "run3_uniform", Run3EquatorSampler, Run3PoleSampler)
    ]
    
    fig_dir = os.path.join(os.path.dirname(__file__), '..', 'figures')
    
    for name, prefix, EqSampler, PoleSampler in scenarios:
        print(f"\n{'='*60}\nRunning {name}\n{'='*60}")
        
        # Equator
        print(f"--> Equator ({name})")
        eq_runner = MonteCarloRunner(
            n_iterations=n_iterations, seed=seed, verbose=False,
            n_workers=n_workers, config=equator_config, sampler_class=EqSampler
        )
        eq_res = eq_runner.run()
        
        # Pole
        print(f"--> Pole ({name})")
        pole_runner = MonteCarloRunner(
            n_iterations=n_iterations, seed=seed+1000, verbose=False,
            n_workers=n_workers, config=pole_config, sampler_class=PoleSampler
        )
        pole_res = pole_runner.run()
        
        # Plot 1: Total Thickness Distributions  (the 1D plots)
        thickness_save_path = os.path.join(fig_dir, f"{prefix}_thickness.png")
        plot_thickness_comparison(eq_res, pole_res, thickness_save_path, f"{name} (500 GW Budget)")
        
        # Plot 2: Conductive vs Convective Structure KDEs (the 2D plots)
        structure_save_path = os.path.join(fig_dir, f"{prefix}_structure.png")
        plot_structure_comparison(eq_res, pole_res, structure_save_path, f"{name} (2.5k runs)")
        print(f"Saved: {structure_save_path}")
        
if __name__ == "__main__":
    mp.freeze_support()
    run_scenarios()
