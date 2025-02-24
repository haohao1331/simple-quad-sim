from pathlib import Path
import pandas as pd
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm

def extract_params(filename : Path):
    ''' Extract wind direction, amplitude, and omega from the filename. '''
    params = str(filename.stem).split('_')
    direction = params[0].replace('dir', '').replace('[', '').replace(']', '')
    amp = float(params[1].replace('amp', ''))
    omega = float(params[2].replace('omega', ''))
    # Convert direction string back to numpy vector
    direction_str = direction.split(' ')
    direction_vec = np.array([float(x) for x in direction_str if x != ''])
    
    return direction_vec, amp, omega

def get_all_wind_directions(dir_path : Path):
    ''' Get all unique wind directions from the trajectory files in the given directory. '''
    all_files = list(dir_path.glob("*trajectory.npy"))
    all_wind_directions = []
    for file in all_files:
        direction_vec, amp, omega = extract_params(file)
        all_wind_directions.append(direction_vec)
    all_wind_directions = np.unique(all_wind_directions, axis=0)
    return all_wind_directions

class TrajectoryDataset(Dataset):
    def __init__(self, data_dir: Path, c):
        self.data_dir = data_dir
        self.trajectory_files = sorted(list(self.data_dir.glob("*trajectory.npy")))
        self.wind_directions = get_all_wind_directions(data_dir)
        self.trajectory_lengths = []
        self.data = pd.DataFrame(columns=['wind_direction_idx', 'amp', 'omega', 'trajectory'])
        for file in tqdm(self.trajectory_files):
            wind_direction, amp, omega = extract_params(file)


            f_wind = np.load(Path(str(file).replace('trajectory', 'F_winds')))
            omega_motor = np.load(Path(str(file).replace('trajectory', 'omegas_motors')))
            states = np.load(file)

            combined_data = np.hstack((f_wind, omega_motor, states))

            # Add row to dataframe
            wind_direction_idx = np.where(np.all(self.wind_directions == wind_direction, axis=1))[0][0]
            self.data = pd.concat([self.data, pd.DataFrame({
                'wind_direction_idx': [wind_direction_idx],
                'amp': [amp], 
                'omega': [omega],
                'trajectory': [combined_data]
            })], ignore_index=True)
            self.trajectory_lengths.append(combined_data.shape[0])
        
        
        self.trajectory_lengths = np.array(self.trajectory_lengths)
        # Verify all trajectories have the same length
        if not np.all(self.trajectory_lengths == self.trajectory_lengths[0]):
            raise ValueError(f"All trajectories must have the same length. Found lengths: {np.unique(self.trajectory_lengths)}")

        self.trajectory_length = self.trajectory_lengths[0]
        del self.trajectory_lengths
        
        self.np_data = np.concatenate(self.data['trajectory'].to_numpy())
        self.X = self.np_data[:, np.r_[3:10, 13:17]]    # corresponds to angular velocity, quarternion, and motor speeds
        self.Y = self.np_data[:, 17:20]     # the x y z component of wind force
        self.c = np.repeat(self.data['amp'].to_numpy(), self.trajectory_length)
        print(f'X.shape: {self.X.shape} Y.shape: {self.Y.shape} c.shape: {self.c.shape}')

        c_idx = np.where(self.c == c)[0]
        self.X = self.X[c_idx]
        self.Y = self.Y[c_idx]
        self.c = self.c[c_idx]

        del self.data

    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.c[idx]
    
if __name__ == "__main__":
    data_dir = Path('/Users/yefan/Desktop/CDS245/simple-quad-sim/data')
    dataset = TrajectoryDataset(data_dir)
    # print(len(dataset))
    # print(dataset[0])
    # print(dataset[1])
    # print(dataset[2])
    # print(dataset[3])