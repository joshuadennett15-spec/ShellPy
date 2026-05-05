"""Diagnose polar rejection — inline diagnostic with same API as MC pipeline."""
import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from regional_samplers import PoleParameterSampler
from Monte_Carlo import SolverConfig
from Solver import Thermal_Solver
from constants import Thermal, Planetary, Convection as CC
from Boundary_Conditions import FixedTemperature

config = SolverConfig(
    nx=31, initial_thickness=20e3, dt=1e12, total_time=5e14,
    eq_threshold=1e-12, max_steps=1500, use_convection=True,
    rannacher_steps=4, use_warm_start=True,
)

counts = {'total':0, 'solver_crash':0, 'filter1_low':0, 'filter1_high':0, 
          'filter2':0, 'filter4':0, 'valid':0}
crash_reasons = []

N = 100
for i in range(N):
    counts['total'] += 1
    sampler = PoleParameterSampler(seed=42 + i)
    params = sampler.sample()
    T_surf = params['T_surf']
    D_H2O = params['D_H2O']
    P_tidal = params['P_tidal']
    H_rad = params['H_rad']
    
    R_europa = Planetary.RADIUS
    R_rock = R_europa - D_H2O
    A_surface = Planetary.AREA
    rho_rock = 3500.0
    M_rock = (4.0/3.0) * np.pi * (R_rock**3) * rho_rock
    q_radiogenic = (H_rad * M_rock) / A_surface
    q_silicate_tidal = P_tidal / A_surface
    q_basal = q_radiogenic + q_silicate_tidal
    
    T_melt = Thermal.MELT_TEMP
    delta_T = T_melt - T_surf
    k_mean = Thermal.conductivity((T_surf + T_melt) / 2)
    H_guess = (k_mean * delta_T) / q_basal
    if config.use_convection:
        H_guess *= 8.0
    H_guess = np.clip(H_guess, 5e3, 100e3)
    
    try:
        surface_bc = FixedTemperature(temperature=T_surf)
        solver = Thermal_Solver(
            nx=config.nx, thickness=H_guess, dt=config.dt,
            total_time=config.total_time, surface_bc=surface_bc,
            rannacher_steps=config.rannacher_steps,
            use_convection=config.use_convection, physics_params=params,
        )
        for step in range(config.max_steps):
            velocity = solver.solve_step(q_basal)
            if abs(velocity) < config.eq_threshold:
                break
        
        H_km = solver.H / 1000.0
        D_H2O_km = D_H2O / 1000.0
        
        if H_km <= 0.5:
            counts['filter1_low'] += 1
        elif H_km >= D_H2O_km * 0.99:
            counts['filter1_high'] += 1
        elif H_km > 200:
            counts['filter2'] += 1
        elif (config.use_convection and solver.convection_state is not None
              and solver.convection_state.D_conv > 0
              and solver.convection_state.Ra < CC.RA_CRIT):
            counts['filter4'] += 1
        else:
            counts['valid'] += 1
            
    except Exception as e:
        counts['solver_crash'] += 1
        crash_reasons.append(str(e)[:100])

print(f"POLAR DIAGNOSTIC ({N} samples):")
print(f"  q_basal range:  ~{q_basal*1000:.1f} mW/m2")
print(f"  H_guess:        {H_guess/1000:.1f} km")
print()
for k, v in counts.items():
    pct = v / counts['total'] * 100
    print(f"  {k:18s}: {v:4d}  ({pct:5.1f}%)")

if crash_reasons:
    from collections import Counter
    print("\nTop crash reasons:")
    for reason, cnt in Counter(crash_reasons).most_common(5):
        print(f"  [{cnt:3d}x] {reason}")
