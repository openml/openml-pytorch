from types import SimpleNamespace
import pandas as pd
import torch
import pytest
from unittest.mock import MagicMock, Mock
from openml_pytorch.trainer import data_handlers, BaseDataHandler, OpenMLImageHandler, OpenMLTabularHandler, OpenMLImageDataset, OpenMLTabularDataset, DataContainer
from torch.utils.data import DataLoader, TensorDataset

@pytest.fixture
def basehandler():
    return BaseDataHandler()

def test_base_handler_data(basehandler):
    assert basehandler.prepare_data is not None
    assert basehandler.prepare_test_data is not None

def test_data_handlers_dict_not_empty():
    assert len(data_handlers) > 0

@pytest.fixture
def imagehandler():
    return OpenMLImageHandler()

@pytest.fixture
def tabularhandler():
    return OpenMLTabularHandler()

@pytest.fixture
def dataconfig():
    return SimpleNamespace(
        file_dir= "",
        transform = None,
        transform_test = None,
        image_size = 32
    )

class TestImageHandler:
    def test_prepare_data(self, imagehandler, dataconfig):
        self.imagehandler = imagehandler
        self.mock_X = pd.DataFrame({"image_name": ["img1.jpg", "img2.jpg"]})
        self.mock_y = pd.Series([0, 1])
        self.image_size = 64
        self.image_dir = "test/images"
        train,val = self.imagehandler.prepare_data(
            X_train = self.mock_X,
            y_train = self.mock_y,
            X_val = self.mock_X,
            y_val = self.mock_y,
            data_config = dataconfig
        )
        assert isinstance(train, OpenMLImageDataset)
        assert len(train) is not None
        assert isinstance(val, OpenMLImageDataset)
        assert len(val) is not None
    
    def test_prepare_test_data(self, imagehandler, dataconfig):
        self.imagehandler = imagehandler
        self.mock_X = pd.DataFrame({"image_name": ["img1.jpg", "img2.jpg"]})
        self.mock_y = pd.Series([0, 1])
        self.image_size = 64
        self.image_dir = "test/images"
        test = self.imagehandler.prepare_test_data(
            X_test = self.mock_X,
            data_config = dataconfig
        ) 

        assert isinstance(test, OpenMLImageDataset)

        
class TestTabularHandler:
    def test_prepare_data(self, tabularhandler, dataconfig):
        self.tabularhandler = tabularhandler
        self.mock_X = pd.DataFrame({"feature1": ["A", "B", "C"], "feature2": [1, 2, 3]})
        self.mock_y = pd.Series([0, 1, 0])

        train,val = self.tabularhandler.prepare_data(
            X_train = self.mock_X,
            y_train = self.mock_y,
            X_val = self.mock_X,
            y_val = self.mock_y,
            data_config = dataconfig
        )
        assert isinstance(train, OpenMLTabularDataset)
        assert len(train) is not None
        assert isinstance(val, OpenMLTabularDataset)
        assert len(val) is not None
    
    def test_prepare_test_data(self, tabularhandler, dataconfig):
        self.tabularhandler = tabularhandler
        self.mock_X = pd.DataFrame({"feature1": ["A", "B", "C"], "feature2": [1, 2, 3]})
        self.mock_y = pd.Series([0, 1, 0])

        test = self.tabularhandler.prepare_test_data(
            X_test = self.mock_X,
            data_config = dataconfig
        ) 

        assert isinstance(test, OpenMLTabularDataset)

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