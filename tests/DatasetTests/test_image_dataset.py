import unittest
import pandas as pd
import torch
from unittest.mock import patch
import torchvision.transforms as T
from openml_pytorch.custom_datasets import OpenMLImageDataset


# Test class for OpenMLImageDataset
class TestOpenMLImageDataset(unittest.TestCase):
    @patch("torchvision.io.read_image")  # Mock read_image function for testing
    def setUp(self, mock_read_image):
        # Setup mock data
        self.mock_X = pd.DataFrame({"image_name": ["img1.jpg", "img2.jpg"]})
        self.mock_y = pd.Series([0, 1])
        self.image_size = 64
        self.image_dir = "tests/images"

        # Mock the read_image function to return a dummy tensor
        mock_read_image.return_value = torch.zeros(
            (3, self.image_size, self.image_size)
        )

        # Create dataset
        self.dataset = OpenMLImageDataset(
            self.mock_X, self.mock_y, self.image_size, self.image_dir
        )

    def test_initialization(self):
        # Test if the dataset is initialized properly
        self.assertEqual(len(self.dataset), 2)  # Two images in mock data
        self.assertEqual(self.dataset.image_size, 64)
        self.assertEqual(self.dataset.image_dir, "tests/images")

    def test_getitem_with_label(self):
        # Test __getitem__ with label transformation
        image, label = self.dataset[0]  # Get first item
        self.assertEqual(
            image.shape, (3, self.image_size, self.image_size)
        )  # Image shape should be (3, H, W)
        self.assertEqual(label, 0)  # Label should be 0

    def test_getitem_without_label(self):
        # Test __getitem__ without label (set self.mock_y to None)
        self.dataset_no_label = OpenMLImageDataset(
            self.mock_X, None, self.image_size, self.image_dir
        )
        image = self.dataset_no_label[0]
        self.assertEqual(
            image.shape, (3, self.image_size, self.image_size)
        )  # Image shape should be (3, H, W)

    def test_len(self):
        # Test the length of the dataset
        self.assertEqual(len(self.dataset), 2)

    def test_transformations(self):
        # Test if transformations are applied
        transform = T.Compose([T.Resize((128, 128))])
        dataset_with_transform = OpenMLImageDataset(
            self.mock_X,
            self.mock_y,
            self.image_size,
            self.image_dir,
            transform_x=transform,
        )
        image, label = dataset_with_transform[0]
        self.assertEqual(
            image.shape, (3, 128, 128)
        )  # Image should be resized to 128x128
