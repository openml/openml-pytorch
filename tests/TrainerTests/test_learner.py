import unittest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import MagicMock, patch
from collections import OrderedDict
from types import SimpleNamespace
import pytest

from openml_pytorch.trainer import Learner

class DummyDataset(torch.utils.data.Dataset):
    def __getitem__(self, idx):
        return torch.tensor([1.0, 2.0]), torch.tensor([1.0])

    def __len__(self):
        return 3


@pytest.fixture
def learner():
    dummy_model = torch.nn.Sequential(
        torch.nn.Linear(2, 2),
    )
    dummy_opt = torch.optim.Adam(dummy_model.parameters())
    dummy_loss_fn = torch.nn.MSELoss()
    dummy_scheduler = torch.optim.lr_scheduler.StepLR(dummy_opt, step_size=1, gamma=0.1)
    dummy_data = MagicMock()
    dummy_data.train_dl = torch.utils.data.DataLoader(DummyDataset(), batch_size=1)
    dummy_data.valid_dl = torch.utils.data.DataLoader(DummyDataset(), batch_size=1)
    dummy_data.test_dl = torch.utils.data.DataLoader(DummyDataset(), batch_size=1)
    dummy_model_classes = OrderedDict()
    dummy_device = torch.device("cpu")
    return Learner(
        model=dummy_model,
        opt=dummy_opt,
        loss_fn=dummy_loss_fn,
        scheduler=dummy_scheduler,
        data=dummy_data,
        model_classes=dummy_model_classes,
        device=dummy_device
    )

def test_learner_init(learner):
    assert learner.model is not None
    assert learner.opt is not None
    assert learner.loss_fn is not None
    assert learner.scheduler is not None
    assert learner.data is not None
    assert learner.model_classes is not None
    assert learner.device is not None
    assert learner.model_classes == {}
