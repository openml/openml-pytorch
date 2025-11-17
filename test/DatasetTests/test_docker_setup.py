import torch
import openml
import numpy as np
import pytest
from openml_pytorch import GenericDataset
from torch.utils.data import DataLoader

def test_pytorch_setup():
    assert torch.__version__ is not None


def test_openml_connection():
    datasets = openml.datasets.list_datasets(output_format='dataframe')
    assert len(datasets) > 0, "No datasets found in OpenML"
    assert 'did' in datasets.columns
    assert 'name' in datasets.columns

def test_generic_dataset():
    X = np.random.rand(100, 5).astype(np.float32)
    y = np.random.randint(0, 2, size=100).astype(np.int64)
    
    # Create dataset and dataloader
    dataset = GenericDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
    
    # Test one batch
    batch = next(iter(dataloader))
    batch_x, batch_y = batch
    
    # Assertions
    assert isinstance(batch_x, torch.Tensor)
    assert isinstance(batch_y, torch.Tensor)
    assert batch_x.shape == (10, 5)  # batch_size x features
    assert batch_y.shape == (10,)    # batch_size
    assert batch_x.dtype == torch.float32
    assert batch_y.dtype == torch.int64

def test_openml_dataset_loading():
    dataset = openml.datasets.get_dataset(61)  
    X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
    
    # Convert to numpy arrays
    X = X.to_numpy(dtype=np.float32)
    
    # Convert string labels to numeric values
    y = y.astype('category').cat.codes.to_numpy().astype(np.int64)
    
    # Create dataset and test one batch
    dataset = GenericDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
    batch_x, batch_y = next(iter(dataloader))
    
    # Basic assertions
    assert batch_x.shape[0] == 10  # batch size
    assert batch_y.shape[0] == 10
    assert batch_x.shape[1] == X.shape[1]  # number of features
    assert batch_y.dtype == torch.int64
    assert torch.all(batch_y < 3)  # Should be 3 classes in iris dataset