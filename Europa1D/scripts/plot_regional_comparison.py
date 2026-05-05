"""
Plot Regional Comparison: Equator vs Pole Ice Shell Structure

Generates two publication-quality figures:
1. Overlaid thickness distributions for equator vs pole
2. Conductive vs convective thickness scatter plot

Requires: equator_results.npz and pole_results.npz from run_regional_comparison.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from Monte_Carlo import load_results

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')


def plot_thickness_comparison(equator, pole, save_path=None):
    """
    Plot overlaid thickness distributions for equator vs pole.
    
    Args:
        equator: MonteCarloResults for equator
        pole: MonteCarloResults for pole
        save_path: Output file path
    """
    if save_path is None:
        save_path = os.path.join(FIGURES_DIR, "regional_comparison_thickness.png")

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
    ax.set_title("Europa Ice Shell Thickness: Equator vs Pole", fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, None)
    ax.set_ylim(0, None)
    
    # Add annotation box with key stats
    stats_text = (
        f"Equator: {equator.cbe_km:.1f} km [{equator.sigma_1_low_km:.1f}-{equator.sigma_1_high_km:.1f}]\n"
        f"Pole: {pole.cbe_km:.1f} km [{pole.sigma_1_low_km:.1f}-{pole.sigma_1_high_km:.1f}]\n"
        f"Ratio: {equator.cbe_km/pole.cbe_km:.1f}×"
    )
    ax.text(0.98, 0.65, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close(fig)


def plot_structure_comparison(equator, pole, save_path=None):
    """
    Plot conductive vs convective thickness using 2D KDE contours for equator vs pole.
    
    Uses kernel density estimation to show the "core" behavior of each region
    without overplotting issues.
    
    Args:
        equator: MonteCarloResults for equator
        pole: MonteCarloResults for pole
        save_path: Output file path
    """
    if save_path is None:
        save_path = os.path.join(FIGURES_DIR, "regional_comparison_structure.png")

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
        common_norm=False,  # Normalize each region separately
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
            # Label the line at a good position
            label_x = min(H_total * 0.85, max_x * 0.9)
            label_y = H_total - label_x
            if 0 < label_y < max_y * 0.9:
                ax.text(label_x, label_y, f'H={H_total}', fontsize=8, alpha=0.5, 
                        rotation=-45, ha='center', va='center')
    
    # Labels and styling
    ax.set_xlabel("Conductive Lid Thickness, D_cond (km)", fontsize=12)
    ax.set_ylabel("Convective Layer Thickness, D_conv (km)", fontsize=12)
    ax.set_title("Ice Shell Structure: 2D Density of Conductive vs Convective Layer", 
                 fontsize=14, fontweight='bold')
    ax.set_xlim(0, max_x)
    ax.set_ylim(0, max_y)
    ax.grid(True, alpha=0.3, linestyle=':')
    
    # Custom legend (KDE creates its own, we add the means)
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
    print(f"Saved: {save_path}")
    plt.close(fig)


def main():
    """Load results and generate both comparison plots."""
    print("Loading results...")
    
    try:
        equator = load_results(os.path.join(RESULTS_DIR, "equator_results.npz"))
        pole = load_results(os.path.join(RESULTS_DIR, "pole_results.npz"))
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run 'python scripts/run_regional_comparison.py' first to generate results.")
        return
    
    print(f"Equator: {equator.n_valid} samples, CBE = {equator.cbe_km:.1f} km")
    print(f"Pole: {pole.n_valid} samples, CBE = {pole.cbe_km:.1f} km")
    
    # Generate plots
    print("\nGenerating plots...")
    plot_thickness_comparison(equator, pole)
    plot_structure_comparison(equator, pole)
    
    print("\nDone! Generated:")
    print("  - figures/regional_comparison_thickness.png")
    print("  - figures/regional_comparison_structure.png")


if __name__ == "__main__":
    main()
