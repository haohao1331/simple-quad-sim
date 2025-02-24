'''
contains helper functions for visualization, not used in main code
'''

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

def plot_single_trajectory(trajectory_file, ref_trajectory, plots_dir):
    # Load trajectory data
    trajectory = np.load(trajectory_file)
    
    # Extract parameters from filename
    params = str(trajectory_file.stem).split('_')
    direction = params[0].replace('dir', '').replace('[', '').replace(']', '')
    amp = float(params[1].replace('amp', ''))
    omega = float(params[2].replace('omega', ''))
    
    # Create figure
    plt.figure(figsize=(10, 8))
    plt.clf()
    
    # Plot trajectory and reference
    plt.plot(trajectory[:,0], trajectory[:,1], 'b-', linewidth=1, label='Actual')
    plt.plot(ref_trajectory[:,0], ref_trajectory[:,1], 'r--', 
            linewidth=2, label='Reference')
    
    # Set title and labels
    plt.title(f'Wind Direction={direction}\nAmplitude={amp}, ω={omega}')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.grid(True)
    plt.axis('equal')
    plt.legend()
    
    # Save plot
    plot_filename = trajectory_file.stem.replace('trajectory', 'plot') + '.png'
    plt.savefig(plots_dir / plot_filename, dpi=300, bbox_inches='tight')
    plt.close()

def plot_all_trajectories():
    # Define the directories
    data_dir = Path('/Users/yefan/Desktop/CDS245/simple-quad-sim/data')
    
    # Create plots directory if it doesn't exist
    plots_dir = Path('/Users/yefan/Desktop/CDS245/simple-quad-sim/plots')
    plots_dir.mkdir(exist_ok=True)
    
    # Get all trajectory files
    trajectory_files = sorted(list(data_dir.glob("*trajectory.npy")))
    print(f"Found {len(trajectory_files)} trajectory files")
    
    # Create reference trajectory
    T = 3
    sim_time = 20.0
    dt = 1.0 / 200
    r = 2*np.pi * np.linspace(0, sim_time, int(sim_time / dt)) / T
    ref_trajectory = np.vstack([np.cos(r/2 - np.pi/2), np.sin(r), np.zeros(len(r))]).T
    
    # Create individual plots for each trajectory
    print(f"Generating {len(trajectory_files)} plots...")
    for file in tqdm(trajectory_files):
        plot_single_trajectory(file, ref_trajectory, plots_dir)
        
    
    print(f"Plots saved to {plots_dir}")

def plot_state_distributions(data_dir):
    """Plot histograms of angular velocities and motor speeds for different wind conditions"""
    # Get all trajectory and motor files
    trajectory_files = list(data_dir.glob("*trajectory.npy"))
    motor_files = list(data_dir.glob("*omegas_motors.npy"))
    
    # Get unique amplitudes and omegas
    amplitudes = sorted(list(set(
        float(f.stem.split('_')[1].replace('amp', '')) 
        for f in trajectory_files
    )))
    wind_omegas = sorted(list(set(
        float(f.stem.split('_')[2].replace('omega', '')) 
        for f in trajectory_files
    )))
    
    # Create subplots for angular velocities and motor speeds
    fig1, axes1 = plt.subplots(len(amplitudes), 3, 
                              figsize=(15, 5*len(amplitudes)))
    fig1.suptitle('Angular Velocity Distributions', fontsize=16)
    
    fig2, axes2 = plt.subplots(len(amplitudes), 4, 
                              figsize=(20, 5*len(amplitudes)))
    fig2.suptitle('Motor Speed Distributions', fontsize=16)
    
    # Plot distributions for each amplitude
    for i, amp in enumerate(amplitudes):
        ang_vel_data = {
            'ω_x': [],
            'ω_y': [],
            'ω_z': []
        }
        motor_data = {
            'Motor 1': [],
            'Motor 2': [],
            'Motor 3': [],
            'Motor 4': []
        }
        
        # Collect data for this amplitude
        for traj_file, motor_file in zip(trajectory_files, motor_files):
            if f'amp{amp}_' in traj_file.name:
                trajectory = np.load(traj_file)
                motors = np.load(motor_file)
                
                # Get angular velocities (last 3 components of state)
                ang_vel_data['ω_x'].extend(trajectory[:, 10])
                ang_vel_data['ω_y'].extend(trajectory[:, 11])
                ang_vel_data['ω_z'].extend(trajectory[:, 12])
                
                # Get motor speeds
                for j in range(4):
                    motor_data[f'Motor {j+1}'].extend(motors[:, j])
        
        # Plot angular velocity distributions
        for j, (key, data) in enumerate(ang_vel_data.items()):
            axes1[i, j].hist(data, bins=50, density=True, alpha=0.7)
            axes1[i, j].set_title(f'{key} (Amp={amp})')
            axes1[i, j].set_xlabel('Angular Velocity (rad/s)')
            axes1[i, j].set_ylabel('Density')
            axes1[i, j].grid(True)
        
        # Plot motor speed distributions
        for j, (key, data) in enumerate(motor_data.items()):
            axes2[i, j].hist(data, bins=50, density=True, alpha=0.7)
            axes2[i, j].set_title(f'{key} (Amp={amp})')
            axes2[i, j].set_xlabel('Motor Speed')
            axes2[i, j].set_ylabel('Density')
            axes2[i, j].grid(True)
    
    # Save plots
    plots_dir = Path('/Users/yefan/Desktop/CDS245/simple-quad-sim/plots_histogram')
    plots_dir.mkdir(exist_ok=True)
    fig1.tight_layout()
    fig2.tight_layout()
    fig1.savefig(plots_dir / 'angular_velocity_distributions.png', dpi=300, bbox_inches='tight')
    fig2.savefig(plots_dir / 'motor_speed_distributions.png', dpi=300, bbox_inches='tight')
    plt.close('all')
    
    print(f"Distribution plots saved to {plots_dir}")

def main():
    data_dir = Path('/Users/yefan/Desktop/CDS245/simple-quad-sim/data')
    plot_state_distributions(data_dir)

if __name__ == "__main__":
    main()

# if __name__ == "__main__":
#     plot_all_trajectories() 