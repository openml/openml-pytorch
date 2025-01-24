# TODO: remove this somehow
import logging

from .trainer import OpenMLDataModule, OpenMLTrainerModule

data_module: OpenMLDataModule = OpenMLDataModule()
trainer: OpenMLTrainerModule = OpenMLTrainerModule(
    experiment_name="default", data_module=data_module
)

global logger
# logger is the default logger for the PyTorch extension
logger = logging.getLogger(__name__)  # type: logging.Logger
