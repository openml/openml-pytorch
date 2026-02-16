import unittest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import MagicMock, patch
from collections import OrderedDict
from types import SimpleNamespace
import pytest

from openml_pytorch.trainer import ModelRunner, Learner


class MockCallback:
    _order = 0
    name = "mock_cb"

    def __init__(self):
        self.calls = []

    def set_runner(self, runner):
        self.runner = runner

    def __call__(self, cb_name):
        self.calls.append(cb_name)
        return False


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, x):
        return self.linear(x)


class DummyDataset(torch.utils.data.Dataset):
    def __getitem__(self, idx):
        return torch.tensor([1.0, 2.0]), torch.tensor([1.0])

    def __len__(self):
        return 3


class DummyData:
    def __init__(self):
        self.train_dl = torch.utils.data.DataLoader(DummyDataset(), batch_size=1)
        self.valid_dl = torch.utils.data.DataLoader(DummyDataset(), batch_size=1)

@pytest.fixture 
def dummy_train_dl():
    return torch.utils.data.DataLoader(DummyDataset(), batch_size=1)


class DummyCallback:
    _order = 0

    def set_runner(self, runner):
        self.runner = runner

    def __call__(self, cb_name):
        return False

    @property
    def name(self):
        return "dummy_cb"


class DummyTrainEvalCallback(DummyCallback):
    def __call__(self, cb_name):
        return False


# Patch this in your test module if TrainEvalCallback is used
@pytest.fixture(autouse=True)
def patch_TrainEvalCallback(monkeypatch):
    monkeypatch.setattr(
        "openml_pytorch.trainer.TrainEvalCallback", DummyTrainEvalCallback
    )


@pytest.fixture
def learner():
    model = DummyModel()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()
    scheduler = None
    data = DummyData()
    model_classes = ["class_0"]
    return Learner(model, opt, loss_fn, scheduler, data, model_classes)


def test_runner_initializes_callbacks():
    runner = ModelRunner(cb_funcs=[lambda: DummyCallback()])
    runner.in_train = True
    assert hasattr(runner, "dummy_cb")
    assert isinstance(runner.dummy_cb, DummyCallback)


def test_runner_one_batch_trains(learner):
    runner = ModelRunner()
    runner.fit(1, learner)
    # should run without error and complete the loop
    assert runner.epoch == 0
    assert runner.current_epoch == 1


def test_runner_calls_callbacks(monkeypatch, learner):

    cb = MockCallback()
    runner = ModelRunner(cbs=[cb])
    runner.fit(1, learner)

    assert "begin_fit" in cb.calls
    assert "after_fit" in cb.calls

def test_all_batches_trains(learner, dummy_train_dl):
    runner = ModelRunner()
    runner.learn = learner
    runner.all_batches(dummy_train_dl)

    assert runner.epoch == 0