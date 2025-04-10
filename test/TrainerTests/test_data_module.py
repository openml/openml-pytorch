import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from openml_pytorch.trainer import OpenMLDataModule, DataContainer  # Adjust as needed

@pytest.fixture
def dummy_data():
    X = pd.DataFrame({
        "Filename": [f"img_{i}.png" for i in range(100)],
        "feature1": np.random.randn(100)
    })
    y = pd.Series(np.random.choice(["cat", "dog"], size=100))
    return X, y


@pytest.fixture
def dummy_test_data():
    return pd.DataFrame({
        "Filename": [f"img_{i}.png" for i in range(20)],
        "feature1": np.random.randn(20)
    })


@pytest.fixture
def mock_handler():
    handler = MagicMock()
    handler.prepare_data.return_value = ("train_dataset", "val_dataset")
    handler.prepare_test_data.return_value = "test_dataset"
    return handler


@pytest.fixture
def mock_config(monkeypatch):
    mock = MagicMock()
    mock.return_data_config.return_value = MagicMock(
        validation_split=0.2,
        batch_size=32,
        type_of_data="image",
        filename_col="Filename",
        file_dir="images",
        target_mode="categorical",
        target_column="encoded_labels"
    )
    monkeypatch.setattr("openml_pytorch.trainer.DefaultConfigGenerator", lambda: mock)


@pytest.fixture
def patched_module(monkeypatch, mock_handler, mock_config):
    return OpenMLDataModule(type_of_data="image")


def test_init_sets_config_correctly(patched_module):
    assert patched_module.data_config.type_of_data == "image"
    assert patched_module.handler is not None
    assert isinstance(patched_module.filename_col, str)
    assert isinstance(patched_module.file_dir, str)
    assert isinstance(patched_module.target_mode, str)
    assert patched_module.transform is None
    assert patched_module.transform_test is None
    assert isinstance(patched_module.target_column, str)
    assert isinstance(patched_module.num_workers, int)
    assert isinstance(patched_module.batch_size, int)


def test_split_training_data(patched_module, dummy_data):
    X, y = dummy_data
    X_train, X_val, y_train, y_val = patched_module.split_training_data(X, y)
    assert len(X_train) > 0 and len(X_val) > 0
    assert len(X_train) + len(X_val) == len(X)


def test_encode_labels(patched_module, dummy_data):
    _, y = dummy_data
    y_train, y_val = y.iloc[:80], y.iloc[80:]
    y_train_enc, y_val_enc, classes = patched_module.encode_labels(y_train, y_val)
    assert set(y_train_enc.unique()).issubset({0, 1})
    assert set(y_val_enc.unique()).issubset({0, 1})
    assert set(classes) == {"cat", "dog"}


def test_get_data_calls_handlers_correctly(patched_module, dummy_data, dummy_test_data):
    X, y = dummy_data
    X_test = dummy_test_data
    
    # Patch DataLoader to prevent instantiation errors
    with patch("openml_pytorch.trainer.DataLoader", return_value="dataloader"):
        data_container, model_classes = patched_module.get_data(X, y, X_test, task="classification")

    assert isinstance(data_container, DataContainer)
    assert model_classes is not None
    assert data_container.train_dl == "dataloader"
    assert data_container.valid_dl == "dataloader"
    assert data_container.test_dl == "dataloader"