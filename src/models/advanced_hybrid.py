import torch
import torch.nn as nn
from .base import BaseModel
from .transformer import PositionalEncoding

class HAR_AdvancedHybrid(BaseModel):
    def __init__(self, num_classes=6, in_channels=9, cnn_dim=64, lstm_hidden=64, nhead=4, trans_layers=2):
        super(HAR_AdvancedHybrid, self).__init__()
        
        # ---------------------------------------------------------
        # THE STEM: Shared Local Feature Extractor
        # ---------------------------------------------------------
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, cnn_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(cnn_dim),
            nn.ReLU(),
            # MaxPool reduces time dimension from 128 -> 64, making downstream tasks highly efficient
            nn.MaxPool1d(kernel_size=2, stride=2) 
        )
        
        # ---------------------------------------------------------
        # STREAM A: Temporal Sequence (LSTM)
        # ---------------------------------------------------------
        self.lstm = nn.LSTM(
            input_size=cnn_dim, 
            hidden_size=lstm_hidden, 
            num_layers=1, 
            batch_first=True,
            bidirectional=True # Bi-directional helps with sequence context!
        )
        # Bi-LSTM outputs 2 * hidden_size
        self.lstm_out_dim = lstm_hidden * 2 
        
        # ---------------------------------------------------------
        # STREAM B: Global Attention (Transformer)
        # ---------------------------------------------------------
        self.pos_encoder = PositionalEncoding(cnn_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cnn_dim, 
            nhead=nhead, 
            dim_feedforward=128, 
            dropout=0.2, 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=trans_layers)
        
        # ---------------------------------------------------------
        # THE FUSION HEAD
        # ---------------------------------------------------------
        fusion_dim = self.lstm_out_dim + cnn_dim
        
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # Input: [Batch, Time=128, Channels=9]
        
        # 1. SHARED STEM
        x = x.transpose(1, 2) # To [B, 9, 128] for Conv1d
        shared_features = self.stem(x) # Output: [B, 64, 64]
        shared_features = shared_features.transpose(1, 2) # Back to [B, Time=64, Features=64]
        
        # 2. STREAM A: LSTM PATH
        # We only care about the final hidden state for sequence classification
        _, (h_n, _) = self.lstm(shared_features)
        # Concatenate the forward and backward hidden states from the Bi-LSTM
        lstm_context = torch.cat((h_n[0,:,:], h_n[1,:,:]), dim=1) # [B, 128]
        
        # 3. STREAM B: TRANSFORMER PATH
        trans_input = self.pos_encoder(shared_features)
        trans_output = self.transformer(trans_input) # [B, 64, 64]
        # Global Average Pooling across the time dimension to get the overall "posture"
        trans_context = torch.mean(trans_output, dim=1) # [B, 64]
        
        # 4. FUSION & CLASSIFICATION
        # We mathematically combine the "Movement Flow" (LSTM) with "Static Posture" (Transformer)
        fused_vector = torch.cat((lstm_context, trans_context), dim=1) # [B, 192]
        
        logits = self.classifier(fused_vector) # [B, 6]
        return logits