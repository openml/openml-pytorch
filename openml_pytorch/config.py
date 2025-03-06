# TODO: remove this somehow
import logging

from .trainer import Trainer, DataModule

# data_module: OpenMLDataModule = OpenMLDataModule()
# trainer: OpenMLTrainerModule = OpenMLTrainerModule(
#     experiment_name="default", data_module=data_module, model= None, task_type = None
# )
# data_module = DataModule(batch_size=32, num_workers=4, target_mode="classification")
trainer = Trainer

global logger
# logger is the default logger for the PyTorch extension
logger = logging.getLogger(__name__)  # type: logging.Logger
