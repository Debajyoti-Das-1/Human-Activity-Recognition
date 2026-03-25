import os
import time
import argparse
import torch
import numpy as np
from collections import deque
from src.models.cnn import HAR_1DCNN
from src.models.lstm import HAR_LSTM
from src.models.cnn_lstm import HAR_CNNLSTM
from src.models.transformer import HAR_Transformer
from src.models.advanced_hybrid import HAR_AdvancedHybrid

class RealTimeEngine:
    def __init__(self, model_name, device, window_size=128, num_features=9):
        self.device = device
        self.window_size = window_size
        self.num_features = num_features
        self.model_name = model_name
        
        print(f"[Engine] Booting {self.model_name.upper()} on {self.device}...")
        
        # Dynamic Model Loading
        if model_name == 'cnn':
            self.model = HAR_1DCNN(num_classes=6, in_channels=num_features)
        elif model_name == 'lstm':
            self.model = HAR_LSTM(input_dim=num_features, hidden_dim=64, num_layers=2, num_classes=6)
        elif model_name == 'cnn_lstm':
            self.model = HAR_CNNLSTM(num_classes=6, in_channels=num_features)
        elif model_name == 'transformer':
            self.model = HAR_Transformer(num_classes=6, in_channels=num_features)
        elif model_name == 'advanced_hybrid':
            self.model = HAR_AdvancedHybrid(num_classes=6, in_channels=num_features)

        self.model = self.model.to(self.device)
        model_path = f'experiments/checkpoints/{model_name}/best_model.pth'
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Missing weights at {model_path}. Train the model first.")
            
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval() 
        
        self.buffer = deque(maxlen=self.window_size)
        self.class_names = ['Walking', 'Walking Upstairs', 'Walking Downstairs', 
                            'Sitting', 'Standing', 'Laying']

    def process_tick(self, sensor_tick):
        self.buffer.append(sensor_tick)
        if len(self.buffer) < self.window_size:
            return None
        return self._predict()

    def _predict(self):
        current_window = np.array(self.buffer)
        tensor_window = torch.tensor(current_window, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.model(tensor_window)
            prediction_idx = torch.argmax(logits, dim=1).item()
            confidence = torch.softmax(logits, dim=1).max().item()
            
        return self.class_names[prediction_idx], confidence

def simulate_data_stream():
    parser = argparse.ArgumentParser(description="Live HAR Inference Simulator")
    parser.add_argument('--model', type=str, default='advanced_hybrid', 
                        choices=['cnn', 'lstm', 'cnn_lstm', 'transformer', 'advanced_hybrid'])
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
    engine = RealTimeEngine(model_name=args.model, device=device)
    
    print("[Stream] Connecting to sensor data feed...")
    X_test = np.load('data/processed/X_test.npy')
    y_test = np.load('data/processed/y_test.npy')
    
    walking_idx = np.where(y_test == 1)[0][0]
    sitting_idx = np.where(y_test == 4)[0][0]
    
    stream_data = np.vstack((X_test[walking_idx], X_test[sitting_idx]))
    
    print(f"\n[Stream] LIVE INFERENCE STARTED (50Hz) | Model: {args.model.upper()}")
    print("="*60)
    
    ticks_processed = 0
    start_time = time.time()
    
    for tick in stream_data:
        time.sleep(0.02) 
        ticks_processed += 1
        result = engine.process_tick(tick)
        
        if result and ticks_processed % 10 == 0:
            activity, confidence = result
            print(f"Timestamp: {ticks_processed * 20}ms | Prediction: {activity:<18} | Confidence: {confidence:.2%}")

    total_time = time.time() - start_time
    print("="*60)
    print(f"[Stream] Disconnected. Processed {ticks_processed} ticks in {total_time:.2f} seconds.")

if __name__ == "__main__":
    simulate_data_stream()