import unittest
import torch
from unittest.mock import MagicMock
from openml_pytorch.callbacks import PutDataOnDeviceCallback


class TestPutDataOnDeviceCallback(unittest.TestCase):

    def setUp(self):
        # Setup for testing, creating a mock model and runner
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Creating mock data
        self.xb = torch.randn(2, 3)  # Batch of 2, 3 features
        self.yb = torch.randint(0, 2, (2,))  # Batch of 2, binary targets
        self.pred = torch.randn(2, 3)  # Batch of predictions

        # Mocking the runner object
        self.run = MagicMock()
        self.run.xb = self.xb
        self.run.yb = self.yb
        self.run.pred = self.pred

        # Create the PutDataOnDeviceCallback instance
        self.callback = PutDataOnDeviceCallback(self.device)
        self.callback.set_runner(self.run)

    def test_begin_fit(self):
        # Test that the model is moved to the correct device in begin_fit
        self.callback.model = MagicMock()  # Mock the model
        self.callback.begin_fit()

        # Check if the model was moved to the device
        self.callback.model.to.assert_called_with(self.device)

    def test_begin_batch(self):
        # Test that the data is moved to the correct device in begin_batch
        self.callback.begin_batch()

        # Check that xb and yb are moved to the correct device
        self.assertEqual(self.run.xb.device, self.device)
        self.assertEqual(self.run.yb.device, self.device)

    def test_after_pred(self):
        # Test that the predictions and labels are moved to the correct device in after_pred
        self.callback.after_pred()

        # Check that pred and yb are moved to the device
        self.assertEqual(self.run.pred.device, self.device)
        self.assertEqual(self.run.yb.device, self.device)
