import openml
import pytest
from unittest.mock import MagicMock, patch, create_autospec
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from openml_pytorch.trainer import Learner, OpenMLTrainerModule
import torch
import numpy as np

@pytest.fixture
def dummy_data_module():
    mock_dm = MagicMock()
    mock_dm.data_config.__dict__ = {"device": torch.device("cpu")}
    mock_dm.get_data.return_value = (MagicMock(), [0, 1, 2])
    return mock_dm

def test_init_sets_correct_config(dummy_data_module):
    trainer = OpenMLTrainerModule(
        experiment_name="test_exp",
        data_module=dummy_data_module,
        use_tensorboard=False
    )
    assert trainer.experiment_name == "test_exp"
    assert hasattr(trainer.config, "opt")
    assert trainer.tensorboard_writer is None

def test_prediction_to_probabilities_valid_input():
    trainer = OpenMLTrainerModule("test", MagicMock(), use_tensorboard=False)
    y_pred = [0, 1, 1, 0]
    classes = [0, 1]
    probs = trainer._prediction_to_probabilities(y_pred, classes)
    assert probs.shape == (4, 2)
    assert (probs.sum(axis=1) == 1).all()

def test_prediction_to_probabilities_raises_on_nonlist_classes():
    trainer = OpenMLTrainerModule("test", MagicMock(), use_tensorboard=False)
    with pytest.raises(ValueError):
        trainer._prediction_to_probabilities([0, 1], np.array([0, 1]))

def test_add_callbacks_adds_custom_callbacks(dummy_data_module):
    class CustomCallback:
        def __init__(self): self._order = 1
    cb = CustomCallback()
    trainer = OpenMLTrainerModule("test", dummy_data_module, callbacks=[cb], use_tensorboard=False)
    assert cb in trainer.cbfs
