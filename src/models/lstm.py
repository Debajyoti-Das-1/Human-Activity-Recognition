import torch
import torch.nn as nn
from .base import BaseModel

class HAR_LSTM(BaseModel):
    def __init__(self, input_dim=9, hidden_dim=64, num_layers=2, num_classes=6, dropout_rate=0.5):
        super(HAR_LSTM, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # --- The Recurrent Core ---
        # input_dim: 9 sensor channels
        # hidden_dim: The size of the memory state vector
        # batch_first=True: Expects [Batch, Time, Features]
        self.lstm = nn.LSTM(
            input_size=input_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0 
        )
        
        # --- Classification Head ---
        # We use Dropout to prevent the LSTM from memorizing specific training sequences
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        """
        Input x shape: [Batch, 128, 9]
        """
        # Pass the entire sequence into the LSTM.
        # 'out' contains the hidden states for ALL 128 time steps.
        # 'hn' is the final hidden state for the very last time step.
        out, (hn, cn) = self.lstm(x)
        
        # out shape: [Batch, 128, 64]
        
        # We only care about the network's understanding at the VERY END of the 2.5s window.
        # So we slice out the last time step's output.
        last_time_step_out = out[:, -1, :] 
        # last_time_step_out shape: [Batch, 64]
        
        # Pass through the classification head
        x = self.dropout(last_time_step_out)
        logits = self.fc(x)
        
        return logits