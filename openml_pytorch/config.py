import logging

import torch.nn
import torch.nn.functional
import torch.optim

from openml import OpenMLTask, OpenMLClassificationTask, OpenMLRegressionTask

from typing import Any, Callable

# logger is the default logger for the PyTorch extension
logger = logging.getLogger(__name__)  # type: logging.Logger


# _default_criterion_gen returns a criterion based on the task type - regressions use
# torch.nn.SmoothL1Loss while classifications use torch.nn.CrossEntropyLoss
def _default_criterion_gen(task: OpenMLTask) -> torch.nn.Module:
    if isinstance(task, OpenMLRegressionTask):
        return torch.nn.SmoothL1Loss()
    elif isinstance(task, OpenMLClassificationTask):
        return torch.nn.CrossEntropyLoss()
    else:
        raise ValueError(task)


# criterion_gen returns the criterion based on the task type
criterion_gen = _default_criterion_gen  # type: Callable[[OpenMLTask], torch.nn.Module]


# _default_optimizer_gen returns the torch.optim.Adam optimizer for the given model
def _default_optimizer_gen(model: torch.nn.Module, _: OpenMLTask) -> torch.optim.Optimizer:
    return torch.optim.Adam(params=model.parameters())


# optimizer_gen returns the optimizer to be used for a given torch.nn.Module
optimizer_gen = _default_optimizer_gen  \
    # type: Callable[[torch.nn.Module, OpenMLTask], torch.optim.Optimizer]


# _default_scheduler_gen returns the torch.optim.lr_scheduler.ReduceLROnPlateau scheduler
# for the given optimizer
def _default_scheduler_gen(optim: torch.optim.Optimizer, _: OpenMLTask) -> Any:
    return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optim)


# scheduler_gen the scheduler to be used for a given torch.optim.Optimizer
scheduler_gen = _default_scheduler_gen  # type: Callable[[torch.optim.Optimizer, OpenMLTask], Any]

# batch_size represents the processing batch size for training
batch_size = 64  # type: int

# epoch_count represents the number of epochs the model should be trained for
epoch_count = 32  # type: int


# _default_predict turns the outputs into predictions by returning the argmax of the output tensor
# for classification tasks, and by flattening the prediction in case of the regression
def _default_predict(output: torch.Tensor, task: OpenMLTask) -> torch.Tensor:
    output_axis = output.dim() - 1
    if isinstance(task, OpenMLClassificationTask):
        output = torch.argmax(output, dim=output_axis)
    elif isinstance(task, OpenMLRegressionTask):
        output = output.view(-1)
    else:
        raise ValueError(task)
    return output


# predict turns the outputs of the model into actual predictions
predict = _default_predict  # type: Callable[[torch.Tensor, OpenMLTask], torch.Tensor]


# _default_predict_proba turns the outputs into probabilities using softmax
def _default_predict_proba(output: torch.Tensor) -> torch.Tensor:
    output_axis = output.dim() - 1
    output = output.softmax(dim=output_axis)
    return output


# predict_proba turns the outputs of the model into probabilities for each class
predict_proba = _default_predict_proba  # type: Callable[[torch.Tensor], torch.Tensor]


# _default sanitizer replaces NaNs with 1e-6
def _default_sanitize(tensor: torch.Tensor) -> torch.Tensor:
    tensor = torch.where(torch.isnan(tensor), torch.ones_like(tensor) * torch.tensor(1e-6), tensor)
    return tensor


# sanitize sanitizes the input data in order to ensure that models can be
# trained safely
sanitize = _default_sanitize  # type: Callable[[torch.Tensor], torch.Tensor]


# _default_retype_labels turns the labels into torch.(cuda)LongTensor if the task is classification
# or torch.(cuda)FloatTensor if the task is regression
def _default_retype_labels(tensor: torch.Tensor, task: OpenMLTask) -> torch.Tensor:
    if isinstance(task, OpenMLClassificationTask):
        return tensor.long()
    elif isinstance(task, OpenMLRegressionTask):
        return tensor.float()
    else:
        raise ValueError(task)


# retype_labels changes the types of the labels in order to ensure type compatibility
retype_labels = _default_retype_labels  # type: Callable[[torch.Tensor, OpenMLTask], torch.Tensor]


# _default_progress_callback reports the current fold, rep, epoch, step and loss for every
# training iteration to the default logger
def _default_progress_callback(fold: int, rep: int, epoch: int,
                               step: int, loss: float, accuracy: float):
    logger.info('[%d, %d, %d, %d] loss: %.4f, accuracy: %.4f' %
                (fold, rep, epoch, step, loss, accuracy))


# progress_callback is called when a training step is finished, in order to
# report the current progress
progress_callback = _default_progress_callback  \
    # type: Callable[[int, int, int, int, float, float], None]


def _setup():
    global logger
    global criterion_gen
    global optimizer_gen
    global scheduler_gen
    global batch_size
    global epoch_count
    global predict
    global predict_proba
    global sanitize
    global retype_labels
    global progress_callback


_setup()
