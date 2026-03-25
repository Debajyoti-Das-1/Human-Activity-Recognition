import torch
import torch.nn as nn
import math
from .base import BaseModel

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class HAR_Transformer(BaseModel):
    def __init__(self, num_classes=6, in_channels=9, d_model=64, nhead=8, num_layers=3, dim_feedforward=128, dropout=0.1):
        super(HAR_Transformer, self).__init__()
        
        # 1. Linear Projection: Map 9 sensor features to d_model (64)
        self.input_projection = nn.Linear(in_channels, d_model)
        
        # 2. Positional Encoding: Give the model a sense of "Time"
        self.pos_encoder = PositionalEncoding(d_model)
        
        # 3. Transformer Encoder Layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=True # Essential for our [B, T, C] shape
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 4. Classification Head
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x shape: [Batch, Time=128, Channels=9]
        
        # Project to d_model space
        x = self.input_projection(x) # [B, 128, 64]
        
        # Add temporal context
        x = self.pos_encoder(x)
        
        # Self-Attention blocks
        x = self.transformer_encoder(x) # [B, 128, 64]
        
        # Pooling: We take the mean across the time dimension
        # (Alternatively, we could use a [CLS] token like BERT)
        x = torch.mean(x, dim=1) # [B, 64]
        
        x = self.dropout(x)
        logits = self.fc(x)
        return logits