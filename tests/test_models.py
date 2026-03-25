import unittest
import torch
from src.models.cnn import HAR_1DCNN
from src.models.lstm import HAR_LSTM
from src.models.cnn_lstm import HAR_CNNLSTM

class TestArchitectures(unittest.TestCase):
    def setUp(self):
        # Create a dummy batch of 2 windows: [Batch=2, Time=128, Features=9]
        self.dummy_input = torch.randn(2, 128, 9)
        
    def test_cnn_forward(self):
        model = HAR_1DCNN(num_classes=6, in_channels=9)
        out = model(self.dummy_input)
        self.assertEqual(out.shape, (2, 6), "CNN output shape mismatch")

    def test_lstm_forward(self):
        model = HAR_LSTM(input_dim=9, hidden_dim=64, num_classes=6)
        out = model(self.dummy_input)
        self.assertEqual(out.shape, (2, 6), "LSTM output shape mismatch")

    def test_hybrid_forward(self):
        model = HAR_CNNLSTM(num_classes=6, in_channels=9)
        out = model(self.dummy_input)
        self.assertEqual(out.shape, (2, 6), "Hybrid output shape mismatch")

if __name__ == '__main__':
    unittest.main()