"""
Callbacks module contains classes and functions for handling callback functions during an event-driven process. This makes it easier to customize the behavior of the training loop and add additional functionality to the training process without modifying the core code.

To use a callback, create a class that inherits from the Callback class and implement the necessary methods. Callbacks can be used to perform actions at different stages of the training process, such as at the beginning or end of an epoch, batch, or fitting process. Then pass the callback object to the Trainer.

## How to Use:
```python
trainer = OpenMLTrainerModule(
    data_module=data_module,
    verbose = True,
    epoch_count = 1,
    callbacks=[ <insert your callback class name here> ],
)
```

To add a custom parameter, for example to add a different metric to the AvgStatsCallBack.
```python
trainer = OpenMLTrainerModule(
    data_module=data_module,
    verbose = True,
    epoch_count = 1,
    callbacks=[ AvgStatsCallBack([accuracy]) ],
)

## Useful Callbacks:
- TestCallback: Use when you are testing out new code and want to iterate through the training loop quickly. Stops training after 2 iterations.
"""
import torch
torch.Tensor.ndim = property(lambda x: len(x.shape))

from .helper import listify, camel2snake
from .callback import Callback
from .annealing import annealer, sched_lin, sched_cos, sched_no, sched_exp, combine_scheds, ParamScheduler
from .training_callbacks import TrainEvalCallback, CancelTrainException, CancelEpochException, CancelBatchException, TestCallback
from .recording import Recorder, AvgStats, AvgStatsCallback
from .device_callbacks import PutDataOnDeviceCallback
from .tensorboard import TensorBoardCallback

__all__ = ['Callback', 'annealer', 'sched_lin', 'sched_cos', 'sched_no', 'sched_exp', 'combine_scheds', 'ParamScheduler', 'TrainEvalCallback', 'CancelTrainException', 'CancelEpochException', 'CancelBatchException', 'TestCallback', 'Recorder', 'AvgStats','AvgStatsCallback', 'PutDataOnDeviceCallback', 'TensorBoardCallback', 'listify', 'camel2snake']