import torch
import torch.nn as nn
from .base import BaseModel

class HAR_CNNLSTM(BaseModel):
    def __init__(self, num_classes=6, in_channels=9):
        super(HAR_CNNLSTM, self).__init__()
        
        # --- BLOCK 1: CNN Feature Extractor ---
        # Input shape: [Batch, 9, 128]
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=64, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        # Output: [Batch, 64, 64]
        
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        # Output: [Batch, 128, 32]
        
        # --- BLOCK 2: LSTM Temporal Modeler ---
        # The CNN outputs 128 channels over 32 time steps. 
        # We will permute this to [Batch, Time=32, Features=128] for the LSTM
        self.lstm = nn.LSTM(
            input_size=128, 
            hidden_size=64, 
            num_layers=1, 
            batch_first=True
        )
        
        # --- BLOCK 3: Classification Head ---
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        """
        Input x shape: [Batch, 128, 9] (B, T, C)
        """
        # 1. Transpose for CNN: [B, Time, Channels] -> [B, Channels, Time]
        x = x.transpose(1, 2)
        
        # 2. Pass through CNN (Spatial/Frequency Extraction)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        # At this point, x is [Batch, Channels=128, Time=32]
        
        # 3. Permute for LSTM: [B, Channels, Time] -> [B, Time, Channels]
        # We need the temporal sequence to be the middle dimension again
        x = x.permute(0, 2, 1)
        
        # 4. Pass through LSTM (Chronological Memory)
        out, (hn, cn) = self.lstm(x)
        
        # out shape: [Batch, 32, 64]
        # Slice out the final time step's memory state
        last_out = out[:, -1, :] 
        
        # 5. Classification
        last_out = self.dropout(last_out)
        logits = self.fc(last_out)
        
        return logits