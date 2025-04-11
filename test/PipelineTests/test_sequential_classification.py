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
        
    ############################################################################
    # Define a sequential network that does the initial image reshaping
    # and normalization model.
    processing_net = torch.nn.Sequential(
        op.layers.Functional(function=torch.Tensor.reshape,
                                                    shape=(-1, 1, 28, 28)),
        torch.nn.BatchNorm2d(num_features=1)
    )
    ############################################################################

    ############################################################################
    # Define a sequential network that does the extracts the features from the
    # image.
    features_net = torch.nn.Sequential(
        torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5),
        torch.nn.LeakyReLU(),
        torch.nn.MaxPool2d(kernel_size=2),
        torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5),
        torch.nn.LeakyReLU(),
        torch.nn.MaxPool2d(kernel_size=2),
    )
    ############################################################################

    ############################################################################
    # Define a sequential network that flattens the features and compiles the
    # results into probabilities for each digit.
    results_net = torch.nn.Sequential(
        op.layers.Functional(function=torch.Tensor.reshape,
                                                    shape=(-1, 4 * 4 * 64)),
        torch.nn.Linear(in_features=4 * 4 * 64, out_features=256),
        torch.nn.LeakyReLU(),
        torch.nn.Dropout(),
        torch.nn.Linear(in_features=256, out_features=10),
    )
    ############################################################################
    # openml.config.apikey = 'key'

    ############################################################################
    # The main network, composed of the above specified networks.
    model = torch.nn.Sequential(
        processing_net,
        features_net,
        results_net
    )
    ############################################################################

    return model


@pytest.fixture
def setup_trainer(setup_data_module):
    trainer = op.OpenMLTrainerModule(
        experiment_name= "MNIST",
        data_module=setup_data_module,
        verbose=True,
        epoch_count=1,
        metrics= [accuracy],
        # remove the TestCallback when you are done testing your pipeline. Having it here will make the pipeline run for a very short time.
        callbacks=[
            op.callbacks.TestCallback,
        ],
        opt = torch.optim.Adam,
    )
    op.config.trainer = trainer
    return trainer


@pytest.fixture
def setup_task():
    return openml.tasks.get_task(3573)


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
