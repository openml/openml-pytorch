import json
import os
from pathlib import Path
from typing import Any
from urllib.request import urlretrieve

import openml

import netron

from openml.extensions import register_extension

from . import config, custom_datasets, layers, trainer, extension
from .callbacks import *
from .extension import PytorchExtension
from .metrics import accuracy, accuracy_topk, f1_score
from .trainer import (
    BaseDataHandler,
    DataContainer,
    DefaultConfigGenerator,
    Learner,
    OpenMLDataModule,
    OpenMLImageHandler,
    OpenMLTabularHandler,
    OpenMLTrainerModule,
    convert_to_rgb,
    BasicTrainer,
)
from .custom_datasets import GenericDataset

__all__ = [
    "PytorchExtension",
    "layers",
    "trainer",
    "custom_datasets",
    "config",
    "callbacks",
    "metrics",
    "convert_to_rgb",
    "DefaultConfigGenerator",
    "BaseDataHandler",
    "OpenMLImageHandler",
    "OpenMLTabularHandler",
    "DataContainer",
    "OpenMLDataModule",
    "Learner",
    "OpenMLTrainerModule",
    "accuracy",
    "accuracy_topk",
    "GenericDataset",
    "BasicTrainer",
    "add_experiment_info_to_run",
    "get_onnx_model_from_run_id",
    "add_file_to_run",
]
register_extension(PytorchExtension)


def add_file_to_run(run, file: Any, name: str = "onnx_model") -> None:
    """
    Add a file to the run object. This file will be uploaded to the server
    when the run is published with a name specified by the user.
    """
    # Check if path was provided and return an error
    if isinstance(file, (Path)):
        raise TypeError("Provide the file content instead of the file path.")

    # Save reference to the original method
    original_get_file_elements = run._get_file_elements

    # Add the new file to the file elements dictionary with the specified name
    def modified_get_file_elements():
        elements = original_get_file_elements()  # Call the original method
        elements[name] = (name, file)  # Add the new file
        return elements

    # Override the original method with the modified version
    run._get_file_elements = modified_get_file_elements
    return run


def safe_add(attribute_dict, trainer, attribute_name, key):
    """Helper function to safely add trainer attribute to dictionary with error logging"""
    try:
        attribute_dict[key] = str(getattr(trainer, attribute_name, None))
    except Exception as e:
        print(f"Error adding {key}: {e}")


def add_experiment_info_to_run(run, trainer):
    """
    Add experiment information to the run object
    - experiment_name: Name of the experiment
    - model: Model architecture
    - optimizer: Optimizer used
    - scheduler: Learning rate scheduler used
    - criterion: Loss function used
    - data_config: Data configuration
    - onnx_model: ONNX model file
    - lrs: Learning rates
    - losses: Losses
    """
    experiment_info = {}

    # Mapping of trainer attributes to experiment info keys
    attribute_mapping = {
        "experiment_name": "experiment_name",
        "model": "model",
        "opt": "optimizer",
        "scheds": "scheduler",
        "criterion": "criterion",
        "data_module.data_config": "data_config",
    }

    # Add attributes to experiment info using safe_add function
    for attribute_name, key in attribute_mapping.items():
        safe_add(experiment_info, trainer, attribute_name, key)

    # Add ONNX model info
    run = add_onnx_model_to_run(run, trainer)

    run = add_learning_rates_to_run(run, trainer)
    # Add the experiment info as a JSON file to the run
    try:
        run = add_file_to_run(run, json.dumps(experiment_info), "experiment_info.json")
    except Exception as e:
        print(f"Error adding experiment info to run: {e}")

    # Add losses to the run
    run = add_losses_to_run(run, trainer)

    # Add learning rate plots to the run
    run = add_learning_rate_plots_to_run(run, trainer)

    # Add metric plots to the run
    run = add_metric_plots_to_run(run, trainer)

    # Add loss plots to the run
    run = add_loss_plots_to_run(run, trainer)

    return run


def add_learning_rates_to_run(run, trainer):
    """
    Add learning rates to the run object
    """
    try:
        run = add_file_to_run(run, json.dumps(trainer.lrs), "lrs.json")
    except Exception as e:
        print(f"Error adding lrs to run: {e}")
    return run

def add_learning_rate_plots_to_run(run, trainer):
    """
    Add learning rate plots to the run object
    """
    try:
        # run = add_file_to_run(run, trainer.plot_lr(), "lr.png")
        # the plot_lr function returns a pyplot figure, which is not serializable
        # so we need to save it to a file first
        trainer.plot_lr(save_path="lr.png")
        run = add_file_to_run(run, open("lr.png", "rb"), "lr.png")
        os.unlink("lr.png") # Remove the file after adding it to the run
    except Exception as e:
        print(f"Error adding lrs to run: {e}")
    return run


def add_losses_to_run(run, trainer):
    """
    Add losses to the run object. Converts tensor losses to list of floats
    """
    try:
        tensor_losses = trainer.runner.recorder.losses
        tensor_losses = [loss.item() for loss in tensor_losses]
        run = add_file_to_run(run, json.dumps(tensor_losses), "losses.json")
    except Exception as e:
        print(f"Error adding losses to run: {e}")
    return run


def add_metric_plots_to_run(run, trainer):
    """
    Add metric plots to the run object
    """
    try:
        # run = add_file_to_run(run, trainer.plot_all_metrics(), "metrics.png")
        # the plot_all_metrics function returns a pyplot figure, which is not serializable
        # so we need to save it to a file first
        trainer.plot_all_metrics(save_path="metrics.png")
        run = add_file_to_run(run, open("metrics.png", "rb"), "metrics.png")
        os.unlink("metrics.png") # Remove the file after adding it to the run
    except Exception as e:
        print(f"Error adding metrics to run: {e}")
    return run

def add_loss_plots_to_run(run, trainer):
    """
    Add loss plots to the run object
    """
    try:
        # run = add_file_to_run(run, trainer.plot_loss(), "loss.png")
        # the plot_loss function returns a pyplot figure, which is not serializable
        # so we need to save it to a file first
        trainer.plot_loss(save_path="loss.png")
        run = add_file_to_run(run, open("loss.png", "rb"), "loss.png")
        os.unlink("loss.png") # Remove the file after adding it to the run
    except Exception as e:
        print(f"Error adding loss to run: {e}")
    return run

def add_onnx_model_to_run(run, trainer):
    """
    Add ONNX model to the run object
    """
    try:
        run = add_file_to_run(run, trainer.onnx_model, "model.onnx")
    except:
        print("No ONNX model found")
    return run


def get_onnx_model_from_run_id(run_id):
    run = openml.runs.get_run(run_id)
    url = "https://api.openml.org/data/download/{}/model.onnx".format(
        run.output_files["model_onnx"]
    )
    file_path, _ = urlretrieve(url, "./model.onnx")
    netron.start(file_path, browse=False)
