import unittest
import pandas as pd
import torch
from openml_pytorch.custom_datasets import OpenMLTabularDataset

# Test class for OpenMLTabularDataset
class TestOpenMLTabularDataset(unittest.TestCase):

    def setUp(self):
        # Setup mock data for tabular dataset
        self.mock_X = pd.DataFrame({'feature1': ['A', 'B', 'C'], 'feature2': [1, 2, 3]})
        self.mock_y = pd.Series([0, 1, 0])
        
        # Create dataset
        self.dataset = OpenMLTabularDataset(self.mock_X, self.mock_y)

    def test_initialization(self):
        # Test if the dataset is initialized properly
        self.assertEqual(len(self.dataset), 3)  # Three rows in mock data
        self.assertIn('feature1', self.dataset.data.columns)  # Feature column should be there
        self.assertIn('feature2', self.dataset.data.columns)

    def test_getitem_with_label(self):
        # Test __getitem__ with label
        x, y = self.dataset[0]  # Get first item
        self.assertEqual(x.shape, (2,))  # There are 2 features in mock data
        self.assertEqual(y, 0)  # The label for the first row should be 0

    def test_getitem_without_label(self):
        # Test __getitem__ without label (set self.mock_y to None)
        self.dataset_no_label = OpenMLTabularDataset(self.mock_X, None)
        x = self.dataset_no_label[0]  # Get first item
        self.assertEqual(x.shape, (2,))  # There are 2 features in mock data

    def test_len(self):
        # Test the length of the dataset
        self.assertEqual(len(self.dataset), 3)

    def test_categorical_encoding(self):
        # Test if categorical columns are properly encoded
        self.assertEqual(self.dataset.data['feature1'][0], 0)  # 'A' should be encoded as 0
        self.assertEqual(self.dataset.data['feature1'][1], 1)  # 'B' should be encoded as 1

    def test_tensor_conversion(self):
        # Test if the data is properly converted to tensor format
        x, y = self.dataset[0]
        self.assertIsInstance(x, torch.Tensor)  # x should be a tensor
        self.assertIsInstance(y, torch.Tensor)  # y should be a tensor
