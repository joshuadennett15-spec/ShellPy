"""
Temporary script to update Solver.py with Phase 2 convection tracking.
Run this once and then delete it.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

SRC_DIR = os.path.join(os.path.dirname(__file__), '..', 'src')

# Read the file
solver_path = os.path.join(SRC_DIR, 'Solver.py')
with open(solver_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Define the new run_to_equilibrium method
old_method = '''    def run_to_equilibrium(
        self, q_ocean: float,
        threshold: float = 1e-15,
        max_steps: Optional[int] = None,
        log_interval: int = 100
    ) -> Dict[str, Any]:
        """
        Runs until thermal equilibrium (db/dt → 0).
        Args:
            q_ocean: Ocean heat flux (W/m²)
            threshold: Velocity threshold for equilibrium (m/s)
            max_steps: Maximum steps (None = use total_steps)
            log_interval: Steps between progress logs

        Returns:
            Dictionary with final state and history
        """
        max_steps = max_steps or self.total_steps

        thickness_history = [self.H / 1000]
        velocity_history = []

        for step in range(max_steps):
            velocity = self.solve_step(q_ocean)

            thickness_history.append(self.H / 1000)
            velocity_history.append(velocity)

            if check_equilibrium(velocity, threshold):#
                print(f"\\n[OK] Equilibrium at step {step}")
                print(f"  Thickness: {self.H/1000:.2f} km ")
                print (f"  Velocity: {velocity:.2e} m/s")
                break

            if step % log_interval == 0:
                phase = "BE" if step < self.rannacher_steps else "CN"
                print(f"Step {step:5d} [{phase}]: H = {self.H / 1000:6.2f} km, db/dt = {velocity:+.2e} m/s")

        return {
            'final_thickness_km': self.H / 1000,
            'final_temperature': self.T.copy(),
            'thickness_history': np.array(thickness_history),
            'velocity_history': np.array(velocity_history),
            'steps': step + 1,
            'coordinate_system': self.coordinate_system,
        }'''

new_method = '''    def run_to_equilibrium(
        self, q_ocean: float,
        threshold: float = 1e-15,
        max_steps: Optional[int] = None,
        log_interval: int = 100,
        track_convection: bool = True,
    ) -> Dict[str, Any]:
        """
        Runs until thermal equilibrium (db/dt → 0).
        
        Phase 2 Enhancement: Tracks D_cond and D_conv history when convection enabled.
        
        Args:
            q_ocean: Ocean heat flux (W/m²)
            threshold: Velocity threshold for equilibrium (m/s)
            max_steps: Maximum steps (None = use total_steps)
            log_interval: Steps between progress logs
            track_convection: Whether to record D_cond/D_conv history

        Returns:
            Dictionary with final state and history, including convection diagnostics
        """
        max_steps = max_steps or self.total_steps

        thickness_history = [self.H / 1000]
        velocity_history = []
        
        # Phase 2: Convection tracking history
        D_cond_history = []
        D_conv_history = []
        Ra_history = []
        Nu_history = []

        for step in range(max_steps):
            velocity = self.solve_step(q_ocean)

            thickness_history.append(self.H / 1000)
            velocity_history.append(velocity)
            
            # Phase 2: Track convection state
            if track_convection and self.convection_state is not None:
                D_cond_history.append(self.convection_state.D_cond / 1000.0)
                D_conv_history.append(self.convection_state.D_conv / 1000.0)
                Ra_history.append(self.convection_state.Ra)
                Nu_history.append(self.convection_state.Nu)

            if check_equilibrium(velocity, threshold):
                print(f"\\n[OK] Equilibrium at step {step}")
                print(f"  Thickness: {self.H/1000:.2f} km ")
                print(f"  Velocity: {velocity:.2e} m/s")
                if self.convection_state is not None:
                    print(f"  D_cond: {self.convection_state.D_cond/1000:.2f} km")
                    print(f"  D_conv: {self.convection_state.D_conv/1000:.2f} km")
                    print(f"  Nu: {self.convection_state.Nu:.2f}")
                break

            if step % log_interval == 0:
                phase = "BE" if step < self.rannacher_steps else "CN"
                conv_info = ""
                if self.convection_state is not None and self.convection_state.is_convecting:
                    conv_info = f", Nu={self.convection_state.Nu:.1f}"
                print(f"Step {step:5d} [{phase}]: H = {self.H / 1000:6.2f} km, db/dt = {velocity:+.2e} m/s{conv_info}")

        result = {
            'final_thickness_km': self.H / 1000,
            'final_temperature': self.T.copy(),
            'thickness_history': np.array(thickness_history),
            'velocity_history': np.array(velocity_history),
            'steps': step + 1,
            'coordinate_system': self.coordinate_system,
        }
        
        # Phase 2: Add convection history to results
        if track_convection and D_cond_history:
            result['D_cond_history_km'] = np.array(D_cond_history)
            result['D_conv_history_km'] = np.array(D_conv_history)
            result['Ra_history'] = np.array(Ra_history)
            result['Nu_history'] = np.array(Nu_history)
            
            # Final convection state
            if self.convection_state is not None:
                result['final_D_cond_km'] = self.convection_state.D_cond / 1000.0
                result['final_D_conv_km'] = self.convection_state.D_conv / 1000.0
                result['final_Ra'] = self.convection_state.Ra
                result['final_Nu'] = self.convection_state.Nu
                result['is_convecting'] = self.convection_state.is_convecting
        
        return result'''

if old_method in content:
    content = content.replace(old_method, new_method)
    with open(solver_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print('Successfully updated run_to_equilibrium method')
else:
    print('Could not find old method pattern - file may already be updated or pattern differs')
    # Let's check what we have
    if 'track_convection: bool = True' in content:
        print('File already contains Phase 2 updates!')
    else:
        print('Pattern mismatch - printing context around run_to_equilibrium...')
        idx = content.find('def run_to_equilibrium')
        if idx != -1:
            print(repr(content[idx:idx+500]))
