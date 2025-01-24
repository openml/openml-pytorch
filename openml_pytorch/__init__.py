from .extension import PytorchExtension
from . import config
from . import layers
from . import trainer
from . import custom_datasets
from openml.extensions import register_extension

from .trainer import (
    convert_to_rgb,
    DefaultConfigGenerator,
    BaseDataHandler,
    OpenMLImageHandler,
    OpenMLTabularHandler,
    DataContainer,
    OpenMLDataModule,
    Learner,
    OpenMLTrainerModule,
)

from .metrics import accuracy, accuracy_topk
from .callbacks import *


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


def add_onnx_to_run(run):

    run._old_get_file_elements = run._get_file_elements

    def modified_get_file_elements():
        elements = run._old_get_file_elements()
        elements["onnx_model"] = ("model.onnx", extension.last_models)
        return elements

    run._get_file_elements = modified_get_file_elements
    return run
