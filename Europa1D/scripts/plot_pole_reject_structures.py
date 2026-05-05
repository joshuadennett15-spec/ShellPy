import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
from plot_shell_structure import plot_single

scenarios = [
    ("run_1_polar_heating_pole_reject.npz", "Run 1 Pole (reject subcrit)", "run1_polar_pole_reject_structure.png"),
    ("run_2_eq_heating_pole_reject.npz", "Run 2 Pole (reject subcrit)", "run2_eq_pole_reject_structure.png"),
    ("run_3_uniform_pole_reject.npz", "Run 3 Pole (reject subcrit)", "run3_uniform_pole_reject_structure.png"),
]

for npz_file, label, out_name in scenarios:
    full_path = os.path.join(os.path.dirname(__file__), '..', 'results', npz_file)
    if os.path.exists(full_path):
        plot_single(npz_file, label, out_name)
    else:
        print(f"Not found: {full_path}")
