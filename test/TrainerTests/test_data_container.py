from types import SimpleNamespace
import pandas as pd
import torch
import pytest
from openml_pytorch.trainer import DataContainer
from torch.utils.data import DataLoader, TensorDataset


@pytest.fixture
def dummy_dataloaders():
    # Create dummy datasets
    train_data = TensorDataset(torch.randn(10, 3), torch.randint(0, 2, (10,)))
    val_data = TensorDataset(torch.randn(5, 3), torch.randint(0, 2, (5,)))
    test_data = TensorDataset(torch.randn(3, 3), torch.randint(0, 2, (3,)))

    # Wrap them in DataLoaders
    train_dl = DataLoader(train_data, batch_size=2)
    val_dl = DataLoader(val_data, batch_size=2)
    test_dl = DataLoader(test_data, batch_size=2)

    return train_dl, val_dl, test_dl


def test_data_container_with_test(dummy_dataloaders):
    train_dl, val_dl, test_dl = dummy_dataloaders
    container = DataContainer(train_dl, val_dl, test_dl)

    assert container.train_dl is train_dl
    assert container.valid_dl is val_dl
    assert container.test_dl is test_dl

    assert container.train_ds is train_dl.dataset
    assert container.valid_ds is val_dl.dataset
    assert container.test_ds is test_dl.dataset


def test_data_container_without_test(dummy_dataloaders):
    train_dl, val_dl, _ = dummy_dataloaders
    container = DataContainer(train_dl, val_dl)

    assert container.test_dl is None
    with pytest.raises(AttributeError):
        _ = container.test_ds  # test_ds should raise since test_dl is None
