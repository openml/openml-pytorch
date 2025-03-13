import numpy as np
import openml
from sklearn.calibration import LabelEncoder
from sklearn.model_selection import train_test_split
import torch

class GenericDataset(torch.utils.data.Dataset):
    """
    Generic dataset that takes X,y as input and returns them as tensors"""
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)  # Convert to tensors
        self.y = torch.tensor(y, dtype=torch.long)     # Ensure labels are LongTensor

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
