import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class HARDataset(Dataset):
    """
    PyTorch Dataset for UCI Human Activity Recognition.
    Expects pre-processed numpy arrays of shape (N, 128, 9) and (N, 1).
    """
    def __init__(self, features_path: str, labels_path: str):
        # 1. Load the compiled binary tensors from disk
        self.X = np.load(features_path)
        self.y = np.load(labels_path)
        
        # 2. Mathematical Label Correction
        # UCI HAR labels are 1-indexed (1 to 6). PyTorch requires 0-indexed (0 to 5).
        self.y = self.y - 1 
        
        # The labels are currently shape (N, 1). 
        # CrossEntropyLoss expects a 1D array of shape (N,) for the targets.
        self.y = self.y.squeeze()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        # Neural network weights are float32. 
        # Labels for CrossEntropy classification MUST be long (int64).
        x_tensor = torch.tensor(self.X[idx], dtype=torch.float32)
        y_tensor = torch.tensor(self.y[idx], dtype=torch.long)
        
        return x_tensor, y_tensor

def get_dataloaders(train_x_path, train_y_path, test_x_path, test_y_path, batch_size=64):
    """
    Constructs and returns the DataLoaders for the training engine.
    """
    train_dataset = HARDataset(train_x_path, train_y_path)
    test_dataset = HARDataset(test_x_path, test_y_path)
    
    # SHUFFLE = TRUE for training to break chronological bias
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        drop_last=True # Drops the last incomplete batch to maintain consistent tensor shapes
    )
    
    # SHUFFLE = FALSE for testing. We evaluate sequentially.
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False
    )
    
    return train_loader, test_loader