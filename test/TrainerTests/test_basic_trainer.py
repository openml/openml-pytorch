import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from unittest.mock import MagicMock
from openml_pytorch.trainer import BasicTrainer  # Replace with actual import
import torch.optim as optim


@pytest.fixture
def simple_data():
    X = torch.randn(10, 5)
    y = torch.randn(10, 1)
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=2)


@pytest.fixture
def dummy_model():
    return nn.Linear(5, 1)


@pytest.fixture
def dummy_loss_fn():
    return nn.MSELoss()


@pytest.fixture
def dummy_optimizer():
    return optim.SGD


@pytest.fixture
def trainer(dummy_model, dummy_loss_fn, dummy_optimizer, simple_data):
    return BasicTrainer(
        model=dummy_model,
        loss_fn=dummy_loss_fn,
        opt=dummy_optimizer,
        dataloader_train=simple_data,
        dataloader_test=simple_data,
        device=torch.device("cpu"),
    )


def test_train_step_runs(trainer):
    x, y = next(iter(trainer.dataloader_train))
    loss = trainer.train_step(x, y)
    assert isinstance(loss, float)


def test_test_step_runs(trainer):
    x, y = next(iter(trainer.dataloader_test))
    loss = trainer.test_step(x, y)
    assert isinstance(loss, float)


def test_fit_runs(trainer):
    trainer.fit(epochs=2)
    assert len(trainer.losses["train"]) > 0
    assert len(trainer.losses["test"]) > 0


def test_missing_train_loader_raises(
    dummy_model, dummy_loss_fn, dummy_optimizer, simple_data
):
    trainer = BasicTrainer(
        model=dummy_model,
        loss_fn=dummy_loss_fn,
        opt=dummy_optimizer,
        dataloader_train=None,
        dataloader_test=simple_data,
        device=torch.device("cpu"),
    )
    with pytest.raises(ValueError, match="dataloader_train is not set"):
        trainer.fit(epochs=1)


def test_missing_test_loader_raises(
    dummy_model, dummy_loss_fn, dummy_optimizer, simple_data
):
    trainer = BasicTrainer(
        model=dummy_model,
        loss_fn=dummy_loss_fn,
        opt=dummy_optimizer,
        dataloader_train=simple_data,
        dataloader_test=None,
        device=torch.device("cpu"),
    )
    with pytest.raises(ValueError, match="dataloader_test is not set"):
        trainer.fit(epochs=1)
