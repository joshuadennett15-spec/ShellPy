import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
from plot_shell_structure import plot_single

# The 500 GW script saves these files in the results/ directory:
scenarios = [
    ("run_1_polar_heating_equator.npz", "Run 1 (Polar) Equator", "run1_polar_equator_structure.png"),
    ("run_1_polar_heating_pole.npz", "Run 1 (Polar) Pole", "run1_polar_pole_structure.png"),
    
    ("run_2_eq_heating_equator.npz", "Run 2 (Eq) Equator", "run2_eq_equator_structure.png"),
    ("run_2_eq_heating_pole.npz", "Run 2 (Eq) Pole", "run2_eq_pole_structure.png"),
    
    ("run_3_uniform_equator.npz", "Run 3 (Uniform) Equator", "run3_uniform_equator_structure.png"),
    ("run_3_uniform_pole.npz", "Run 3 (Uniform) Pole", "run3_uniform_pole_structure.png"),
]

def main():
    print("Generating shell structure plots from 500 GW Scenario .npz files...")
    for npz_file, label, out_name in scenarios:
        # Check if file exists since the Monte Carlo run might be still running
        full_path = os.path.join(os.path.dirname(__file__), '..', 'results', npz_file)
        if os.path.exists(full_path):
            try:
                plot_single(npz_file, label, out_name)
                print(f"Successfully generated {out_name}")
            except Exception as e:
                print(f"Error generating {out_name}: {e}")
        else:
            print(f"File not found: {full_path}")

if __name__ == "__main__":
    main()
