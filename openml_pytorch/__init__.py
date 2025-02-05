from typing import Any

from openml.extensions import register_extension

from . import config, custom_datasets, layers, trainer, extension
from .callbacks import *
from .extension import PytorchExtension
from .metrics import accuracy, accuracy_topk
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
)

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

# def add_file_to_run(run, file: Any, name: str = "onnx_model") -> None:
#     """
#     Add a file to the run object. This file will be uploaded to the server
#     when the run is published with a name specified by the user.
#     """
#     # Check if path was provided and return an error

#     if isinstance(file, (Path)):
#         raise TypeError("Provide the file content instead of the file path.")

#     # Save reference to the original method
#     run._old_get_file_elements = run._get_file_elements

#     # Add the new file to the file elements dictionary with the specified name
#     def modified_get_file_elements():
#         onnx_ = extension.last_models
#         elements = run._old_get_file_elements()
#         # elements[name] = (name, file)
#         elements["onnx_model"] = (name, onnx_)
#         return elements

#     # Override the original data
#     run._get_file_elements = modified_get_file_elements
#     return run


def add_onnx_to_run(run):
    run._old_get_file_elements = run._get_file_elements

    def modified_get_file_elements():
        onnx_ = extension.last_models  # saving as local variable to solve RecursionError: maximum recursion depth exceeded
        elements = run._old_get_file_elements()
        elements["onnx_model"] = ("model.onnx", onnx_)
        return elements

    run._get_file_elements = modified_get_file_elements
    return run


# def add_onnx_to_run(run):
#     """
#     Add the last model to the run object in ONNX format.
#     """
#     try:
#         run = add_file_to_run(run, extension.last_models, "model.onnx")
#         return run
#     except Exception as e:
#         print(f"Failed to convert model to ONNX: {e}")

# def add_model_str_to_run(run, model):
#     """
#     Add the model to the run object as a string, for easy comparison.
#     """
#     try:
#         model_str = str(model)
#         run = add_file_to_run(run, model_str, "model_str")
#         return run
#     except Exception as e:
#         print(f"Failed to convert model to string: {e}")
#
# def add_info_to_run(run, trainer):
#     model = trainer.model
#     # Add the model in ONNX format
#     run = add_onnx_to_run(run)
#     # Add the model as a string
#     run = add_model_str_to_run(run, model)
#     # Add experiment name to the run
#     run = add_file_to_run(run, trainer.experiment_name, "experiment_name")
#     return run
