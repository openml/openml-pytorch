from typing import Any

from openml.extensions import register_extension

from . import config, custom_datasets, layers, trainer
from .callbacks import *
from .extension import PytorchExtension
from .metrics import accuracy, accuracy_topk
from .trainer import (BaseDataHandler, DataContainer, DefaultConfigGenerator,
                      Learner, OpenMLDataModule, OpenMLImageHandler,
                      OpenMLTabularHandler, OpenMLTrainerModule,
                      convert_to_rgb)

__all__ = [
    "PytorchExtension",
    "layers",
    "add_onnx_to_run",
    "trainer",
    "custom_datasets",
    "config",
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
]

register_extension(PytorchExtension)


def add_file_to_run(run, file: Any, name: str, max_file_size_mb: int = 100) -> None:
    """
    Add a file to the run object. This file will be uploaded to the server
    when the run is published with a name specified by the user.
    Ensures that the file size is less than 100MB.
    """
    # Check if path was provided and return an error
    if isinstance(file, (str, Path)):
        raise TypeError("Provide the file content instead of the file path.")

    # Check if the file size exceeds the limit
    if not run._check_file_size(file, max_file_size_mb):
        raise ValueError(f"File size exceeds {max_file_size_mb}MB. File: {name}")

    # Save reference to the original method
    run._old_get_file_elements = run._get_file_elements

    # Add the new file to the file elements dictionary with the specified name
    def modified_get_file_elements():
        elements = run._old_get_file_elements()
        elements[name] = (name, file)
        return elements

    # Override the original data
    run._get_file_elements = modified_get_file_elements


def add_onnx_to_run(run):
    add_file_to_run(run, extension.last_models, "model.onnx")
