#TODO: remove this somehow
from .trainer import OpenMLTrainerModule
import logging
trainer: OpenMLTrainerModule = OpenMLTrainerModule()

global logger
# logger is the default logger for the PyTorch extension
logger = logging.getLogger(__name__)  # type: logging.Logger