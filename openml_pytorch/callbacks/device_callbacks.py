from .callback import Callback
import torch
import torch.nn as nn
import torch.distributed as dist


class PutDataOnDeviceCallback(Callback):
    """
    Custom callback to move data and model to the specified device, supporting multi-GPU via DistributedDataParallel (DDP).
    """

    def __init__(self, device):
        self.device = device

    def begin_fit(self):
        """Moves the model to the device and wraps it in DistributedDataParallel if multiple GPUs are available."""
        self.model.to(self.device)
        if torch.cuda.device_count() > 1 and dist.is_initialized():
            self.model = nn.parallel.DistributedDataParallel(
                self.model, device_ids=[self.device]
            )

    def begin_batch(self):
        """Moves input data and target labels to the device."""
        self.run.xb, self.run.yb = self.run.xb.to(
            self.device, non_blocking=True
        ), self.run.yb.to(self.device, non_blocking=True)

    def after_pred(self):
        """Ensures predictions and labels are on the correct device."""
        self.run.pred = self.run.pred.to(self.device)
        self.run.yb = self.run.yb.to(self.device)
