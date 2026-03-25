import torch
import torch.nn as nn
from .base import BaseModel

class HAR_1DCNN(BaseModel):
    def __init__(self, num_classes=6, in_channels=9):
        super(HAR_1DCNN, self).__init__()
        
        # --- BLOCK 1: Feature Extraction ---
        # Input shape: [Batch, 9, 128]
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=64, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        # Output shape after Block 1: [Batch, 64, 64] 
        # (Time reduced from 128 -> 64 due to pooling)
        
        # --- BLOCK 2: Deep Feature Extraction ---
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        # Output shape after Block 2: [Batch, 128, 32]
        # (Time reduced from 64 -> 32)
        
        # --- BLOCK 3: Classification Head ---
        # Flattening [Batch, 128, 32] gives a vector of size 128 * 32 = 4096
        self.flatten = nn.Flatten()
        
        self.fc1 = nn.Linear(128 * 32, 100)
        self.relu3 = nn.ReLU()
        
        # Dropout for Regularization (prevents overfitting to the training sensors)
        self.dropout = nn.Dropout(p=0.5)
        
        # Final output mapping to the 6 activity classes
        self.fc2 = nn.Linear(100, num_classes)

    def forward(self, x):
        """
        The Forward Pass.
        Input x shape from DataLoader: [Batch, 128, 9] (B, T, C)
        """
        # 1. Transpose to [Batch, Channels, Time] -> [B, 9, 128]
        x = x.transpose(1, 2)
        
        # 2. Pass through Conv Block 1
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        # 3. Pass through Conv Block 2
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        # 4. Flatten the tensor
        x = self.flatten(x)
        
        # 5. Pass through Dense Layers
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.dropout(x)
        
        # Raw logits out (No Softmax needed here because PyTorch's 
        # CrossEntropyLoss automatically applies LogSoftmax internally)
        logits = self.fc2(x) 
        
        return logits