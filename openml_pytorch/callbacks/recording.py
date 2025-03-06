from .callback import Callback
import matplotlib.pyplot as plt
from .helper import listify
import torch

class Recorder(Callback):
    """
    Recorder is a callback class used to record learning rates and losses during the training process.
    """

    def begin_fit(self):
        """
        Initializes attributes necessary for the fitting process.

        Sets up learning rates and losses storage.

        Attributes:
            self.lrs (list): A list of lists, where each inner list will hold learning rates for a parameter group.
            self.losses (list): An empty list to store loss values during the fitting process.
        """
        self.lrs = [[] for _ in self.opt.param_groups]
        self.losses = []

    def after_batch(self):
        """
        Handles operations to execute after each training batch.

        Modifies the learning rate for each parameter group in the optimizer
        and appends the current learning rate and loss to the corresponding lists.

        """
        if not self.in_train:
            return
        for pg, lr in zip(self.opt.param_groups, self.lrs):
            lr.append(pg["lr"])
        self.losses.append(self.loss.detach().cpu())

    def plot_lr(self, pgid=-1):
        """
        Plots the learning rate for a given parameter group.
        """
        plot = plt.plot(self.lrs[pgid])
        # self.writer.add_image("Learning rate", plot, 0)
        return plot

    def plot_loss(self, skip_last=0):
        """
        Plots the loss for a given parameter group.
        """
        return plt.plot(self.losses[: len(self.losses) - skip_last])

    def plot(self, skip_last=0, pgid=-1):
        """
        Generates a plot of the loss values against the learning rates.
        """
        losses = [o.item() for o in self.losses]
        lrs = self.lrs[pgid]
        n = len(losses) - skip_last
        plt.xscale("log")
        return plt.plot(lrs[:n], losses[:n])
        # return losses, lrs

class AvgStats:
    """
    AvgStats is a helper class that tracks and averages metrics over an epoch.

    Arguments:
        metrics: A list of metric functions.
        in_train: A boolean indicating whether it's tracking training or validation.
    """
    def __init__(self, metrics, in_train):
        self.metrics = metrics
        self.in_train = in_train
        self.reset()

    def reset(self):
        """Resets stored metric values."""
        self.losses = []
        self.metrics_sums = [0] * len(self.metrics)
        self.count = 0

    def accumulate(self, run):
        """Accumulates loss and metric values from the current batch."""
        bs = run.xb.shape[0]  # Batch size
        self.losses.append(run.loss.item() * bs)
        self.count += bs

        for i, metric in enumerate(self.metrics):
            self.metrics_sums[i] += metric(run.pred, run.yb) * bs

    def __str__(self):
        """Returns formatted string of average loss and metrics."""
        avg_loss = sum(self.losses) / self.count if self.count > 0 else 0
        avg_metrics = [
            metric_sum / self.count if self.count > 0 else 0
            for metric_sum in self.metrics_sums
        ]
        metric_str = " ".join(
            [f"{metric.__name__}: {val:.4f}" for metric, val in zip(self.metrics, avg_metrics)]
        )
        return f"{'Train' if self.in_train else 'Valid'} - Loss: {avg_loss:.4f} {metric_str}"



class AvgStatsCallback(Callback):
    """
    Callback to track and print average loss and metrics for training and validation.

    Arguments:
        metrics: A list of metric functions to evaluate during training and validation.
    """

    def __init__(self, metrics):
        self.train_stats = AvgStats(metrics, True)
        self.valid_stats = AvgStats(metrics, False)

    def begin_epoch(self):
        """Resets stats at the beginning of each epoch."""
        self.train_stats.reset()
        self.valid_stats.reset()

    def after_loss(self):
        """Accumulates metrics after loss calculation."""
        stats = self.train_stats if self.run.in_train else self.valid_stats
        with torch.no_grad():
            stats.accumulate(self.run)

    def after_epoch(self):
        """Prints training and validation statistics at the end of each epoch."""
        print(self.train_stats)
        print(self.valid_stats)