"""100-iteration test of the transient solver with Green et al. convection."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from Solver import Thermal_Solver
from Boundary_Conditions import FixedTemperature

print('=== 100-Iteration Convection Test ===')
print()

# Create solver with Green et al. parameters
solver = Thermal_Solver(
    nx=51,
    thickness=25e3,  # 25 km initial
    dt=1e12,
    use_convection=True,
    surface_bc=FixedTemperature(100.0),
    physics_params={
        'Q_v': 59.4e3,
        'Q_b': 49e3,
        'd_grain': 0.1e-3,  # 0.1 mm grain size
        'epsilon_0': 1e-10,  # Tidal strain (Green et al.)
    }
)

q_ocean = 0.011  # 11 mW/m^2

print(f'Initial thickness: {solver.H/1000:.2f} km')
print(f'Ocean heat flux: {q_ocean*1000:.1f} mW/m^2')
print()
print(f'{"Step":>5} {"H(km)":>8} {"D_cond":>8} {"D_conv":>8} {"Nu":>8} {"Ra":>10} {"db/dt":>12}')
print('-' * 70)

for step in range(100):
    velocity = solver.solve_step(q_ocean)
    
    if step % 10 == 0 or step == 99:
        state = solver.convection_state
        print(f'{step:5d} {solver.H/1000:8.2f} {state.D_cond/1000:8.2f} {state.D_conv/1000:8.2f} '
              f'{state.Nu:8.1f} {state.Ra:10.2e} {velocity:+12.2e}')

print()
print('=== FINAL STATE ===')
print(f'Total thickness: {solver.H/1000:.2f} km')
print(f'Conductive lid: {state.D_cond/1000:.2f} km ({state.D_cond/solver.H*100:.1f}%)')
print(f'Convective layer: {state.D_conv/1000:.2f} km')
print(f'Nusselt number: {state.Nu:.1f}')
print(f'Rayleigh number: {state.Ra:.2e}')
print(f'Convecting: {state.is_convecting}')
print()
print('Temperature Profile (selected nodes):')
depths = solver.nodes * solver.H / 1000
for i in [0, 10, 20, 30, 40, 50]:
    if i < len(solver.T):
        marker = ' <-- Tc' if abs(solver.T[i] - state.T_c) < 5 else ''
        print(f'  z={depths[i]:5.1f} km: T={solver.T[i]:6.1f} K{marker}')
