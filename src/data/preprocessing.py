import os
import numpy as np
import pandas as pd

def load_file(filepath):
    """
    Reads a single .txt file containing space-separated scientific floats.
    Returns a 2D numpy array of shape (Samples, 128).
    """
    dataframe = pd.read_csv(filepath, header=None, sep=r'\s+')
    return dataframe.values

def load_group(filenames, prefix=''):
    """
    Loads a list of 9 files and stacks them along the depth axis.
    """
    loaded = list()
    for name in filenames:
        data = load_file(prefix + name)
        loaded.append(data)
    
    # Stack the 9 arrays of shape (Samples, 128) into (Samples, 128, 9)
    # np.dstack is highly optimized for this specific memory operation
    loaded = np.dstack(loaded)
    return loaded

def compile_dataset(dataset_dir, output_dir):
    """
    Parses the raw UCI HAR structure and saves unified .npy files.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # The 9 sensor channels in the exact order we want to stack them
    signals = [
        'body_acc_x_', 'body_acc_y_', 'body_acc_z_',
        'body_gyro_x_', 'body_gyro_y_', 'body_gyro_z_',
        'total_acc_x_', 'total_acc_y_', 'total_acc_z_'
    ]
    
    for group in ['train', 'test']:
        print(f"[Data] Compiling {group} set...")
        
        # 1. Compile Features (X)
        filenames = [f"{sig}{group}.txt" for sig in signals]
        prefix = os.path.join(dataset_dir, group, 'Inertial Signals', '')
        
        X = load_group(filenames, prefix)
        
        # 2. Compile Labels (y)
        y_path = os.path.join(dataset_dir, group, f'y_{group}.txt')
        y = load_file(y_path)
        
        # Save as fast-loading numpy binaries
        np.save(os.path.join(output_dir, f'X_{group}.npy'), X)
        np.save(os.path.join(output_dir, f'y_{group}.npy'), y)
        
        print(f"       -> X_{group} shape: {X.shape}")
        print(f"       -> y_{group} shape: {y.shape}")

if __name__ == '__main__':
    # FIXED PATHS:
    # Running from the root of har_project, so we don't need '../'
    # And we use the exact folder name with spaces: 'UCI HAR Dataset'
    raw_path = 'data/raw/UCI HAR Dataset'
    processed_path = 'data/processed'
    
    if os.path.exists(raw_path):
        compile_dataset(raw_path, processed_path)
        print("[System] Preprocessing complete. Data ready for PyTorch.")
    else:
        print(f"[Error] Raw data not found at '{raw_path}'. Please check your paths.")