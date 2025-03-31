from openml.extensions import register_extension

from . import config, custom_datasets, layers, trainer
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
    BasicTrainer,
)
from .custom_datasets import GenericDataset
from .run_utils import add_experiment_info_to_run, add_file_to_run, get_onnx_model_from_run_id

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
