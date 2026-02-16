from pathlib import Path
import pytest
import openml
import openml_pytorch as op
import torchvision
import torch
from openml_pytorch.metrics import accuracy
from openml_pytorch.trainer import convert_to_rgb
import matplotlib
matplotlib.use('Agg')  # Must be set before importing pyplot
import matplotlib.pyplot as plt


@pytest.fixture
def setup_data_module():
    data_module = op.OpenMLDataModule(
        type_of_data="dataframe",
        target_column="class",
        target_mode="categorical",
    )
    return data_module


@pytest.fixture
def setup_model():
    class TabularClassificationmodel(torch.nn.Module):
        def __init__(self, input_size, output_size):
            super(TabularClassificationmodel, self).__init__()
            self.fc1 = torch.nn.Linear(input_size, 128)
            self.fc2 = torch.nn.Linear(128, 64)
            self.fc3 = torch.nn.Linear(64, output_size)
            self.relu = torch.nn.ReLU()
            self.softmax = torch.nn.Softmax(dim=1)

        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            x = self.relu(x)
            x = self.fc3(x)
            x = self.softmax(x)
            return x
        
    model = TabularClassificationmodel(20, 2)
    return model


@pytest.fixture
def setup_trainer(setup_data_module):
    trainer = op.OpenMLTrainerModule(
        experiment_name= "Credit-G",
        data_module=setup_data_module,
        verbose=True,
        epoch_count=1,
        metrics= [accuracy],
        # remove the TestCallback when you are done testing your pipeline. Having it here will make the pipeline run for a very short time.
        callbacks=[
            # op.callbacks.TestCallback,
        ],
        opt = torch.optim.Adam,
    )
    op.config.trainer = trainer
    return trainer


@pytest.fixture
def setup_task():
    return openml.tasks.get_task(31)


def test_data_loading(setup_data_module):
    assert setup_data_module is not None


def test_model_initialization(setup_model):
    assert setup_model is not None
    assert isinstance(setup_model, torch.nn.Module)


def test_training_pipeline(setup_model, setup_task, setup_trainer):
    run = openml.runs.run_model_on_task(
        setup_model, setup_task, avoid_duplicate_runs=False
    )
    assert run is not None
    assert setup_trainer.stats.metrics is not None
