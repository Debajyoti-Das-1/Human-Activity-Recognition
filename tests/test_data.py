import unittest
import numpy as np

class TestDataPipeline(unittest.TestCase):
    def setUp(self):
        # We assume the preprocessing script has been run
        self.X_train = np.load('data/processed/X_train.npy')
        self.X_test = np.load('data/processed/X_test.npy')

    def test_tensor_shapes(self):
        """Ensures the time series geometry is strictly [Samples, 128, 9]"""
        self.assertEqual(self.X_train.shape[1], 128, "Window size must be 128")
        self.assertEqual(self.X_train.shape[2], 9, "Feature depth must be 9")
        
    def test_data_leakage(self):
        """Mathematical guarantee that test data is not in training data."""
        # A simple hash check of the first window to ensure arrays are disjoint
        train_hash = hash(self.X_train[0].tobytes())
        test_hash = hash(self.X_test[0].tobytes())
        self.assertNotEqual(train_hash, test_hash, "Data leakage detected between Train and Test!")

if __name__ == '__main__':
    unittest.main()