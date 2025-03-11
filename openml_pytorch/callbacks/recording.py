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
    AvgStats class is used to track and accumulate average statistics (like loss and other metrics) during training and validation phases.

    Attributes:
        metrics (list): A list of metric functions to be tracked.
        in_train (bool): A flag to indicate if the statistics are for the training phase.

    Methods:
        __init__(metrics, in_train):
            Initializes the AvgStats with metrics and in_train flag.

        reset():
            Resets the accumulated statistics.

        all_stats:
            Property that returns all accumulated statistics including loss and metrics.

        avg_stats:
            Property that returns the average of the accumulated statistics.

        accumulate(run):
            Accumulates the statistics using the data from the given run.

        __repr__():
            Returns a string representation of the average statistics.
    """

    def __init__(self, metrics, in_train):
        self.metrics, self.in_train = listify(metrics), in_train

    def reset(self):
        self.tot_loss, self.count = 0.0, 0
        self.tot_mets = [0.0] * len(self.metrics)

    @property
    def all_stats(self):
        return [self.tot_loss.item()] + self.tot_mets

    @property
    def avg_stats(self):
        return [o / self.count for o in self.all_stats]

    def accumulate(self, run):
        bn = run.xb.shape[0]
        self.tot_loss += run.loss * bn
        self.count += bn
        for i, m in enumerate(self.metrics):
            self.tot_mets[i] += m(run.pred, run.yb) * bn

    def __repr__(self):
        if not self.count:
            return ""
        return f"{'train' if self.in_train else 'valid'}: {self.avg_stats}"

class AvgStatsCallback(Callback):
    """
    AvgStatsCallBack class is a custom callback used to track and print average statistics for training and validation phases during the training loop.

    Arguments:
        metrics: A list of metric functions to evaluate during training and validation.

    Methods:
        __init__: Initializes the callback with given metrics and sets up AvgStats objects for both training and validation phases.
        begin_epoch: Resets the statistics at the beginning of each epoch.
        after_loss: Accumulates the metrics after computing the loss, differentiating between training and validation phases.
        after_epoch: Prints the accumulated statistics for both training and validation phases after each epoch.
    """

    def __init__(self, metrics):
        self.train_stats, self.valid_stats = (
            AvgStats(metrics, True),
            AvgStats(metrics, False),
        )

    def begin_epoch(self):
        self.train_stats.reset()
        self.valid_stats.reset()

    def after_loss(self):
        stats = self.train_stats if self.in_train else self.valid_stats
        with torch.no_grad():
            stats.accumulate(self.run)

    def after_epoch(self):
        print(self.train_stats)
        print(self.valid_stats)