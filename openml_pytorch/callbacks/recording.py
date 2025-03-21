from .callback import Callback
import matplotlib.pyplot as plt
from .helper import listify
import torch


class Recorder(Callback):
    """
    Recorder is a callback class used to record learning rates, losses, and metrics during the training process.
    """

    def begin_fit(self):
        """
        Initializes attributes necessary for the fitting process.
        """
        self.lrs = [[] for _ in self.opt.param_groups]
        self.losses = []
        self.metrics = (
            {metric.__name__: [] for metric in self.metrics}
            if hasattr(self, "metrics")
            else {}
        )
        self.epochs = []
        self.current_epoch = 0

    def begin_epoch(self):
        """
        Handles operations at the beginning of each epoch.
        """
        self.current_epoch += 1

    def after_batch(self):
        """
        Handles operations to execute after each training batch.
        """
        if not self.in_train:
            return

        for pg, lr in zip(self.opt.param_groups, self.lrs):
            lr.append(pg["lr"])

        self.losses.append(self.loss.detach().cpu())

    def after_epoch(self):
        """
        Records metrics at the end of each epoch.
        """
        self.epochs.append(self.current_epoch)
        # Record metrics from AvgStatsCallback if available
        if hasattr(self, "run"):
            for cb in self.run.cbs:
                if isinstance(cb, AvgStatsCallback):
                    for i, metric_fn in enumerate(cb.train_stats.metrics):
                        metric_name = metric_fn.__name__
                        if metric_name == "":
                            metric_name = str(metric_fn)
                        if metric_name not in self.metrics:
                            self.metrics[metric_name] = []
                        # Store both train and valid metrics
                        self.metrics[metric_name].append(
                            {
                                "train": cb.train_stats.avg_stats[
                                    i + 1
                                ],  # +1 because avg_stats includes loss as first element
                                "valid": cb.valid_stats.avg_stats[i + 1],
                            }
                        )

    def plot_lr(self, pgid=-1, save_path=None):
        """
        Plots the learning rate for a given parameter group.
        """
        plot = plt.plot(self.lrs[pgid])
        if save_path:
            plt.savefig(save_path)
        return plot

    def plot_loss(self, skip_last=0, save_path=None):
        """
        Plots the loss values.
        """
        plot = plt.plot(self.losses[: len(self.losses) - skip_last])

        if save_path:
            plt.savefig(save_path)
        return plot

    def plot(self, skip_last=0, pgid=-1):
        """
        Generates a plot of the loss values against the learning rates.
        """
        losses = [o.item() for o in self.losses]
        lrs = self.lrs[pgid]
        n = len(losses) - skip_last
        plt.xscale("log")
        return plt.plot(lrs[:n], losses[:n])

    def plot_metric(self, metric_name, skip_last=0, save_path=None):
        """
        Plots a specific metric over epochs.

        Args:
            metric_name (str): Name of the metric to plot
            skip_last (int): Number of last points to skip
        """
        if metric_name not in self.metrics:
            print(
                f"Metric '{metric_name}' not found. Available metrics: {list(self.metrics.keys())}"
            )
            return

        train_vals = [d["train"] for d in self.metrics[metric_name]]
        valid_vals = [d["valid"] for d in self.metrics[metric_name]]

        # convert to cpu numpy if necessary
        train_vals = [
            val.item() if isinstance(val, torch.Tensor) else val for val in train_vals
        ]
        valid_vals = [
            val.item() if isinstance(val, torch.Tensor) else val for val in valid_vals
        ]

        plt.figure(figsize=(10, 6))
        plt.plot(
            self.epochs[:-skip_last] if skip_last > 0 else self.epochs,
            train_vals[:-skip_last] if skip_last > 0 else train_vals,
            label=f"Train {metric_name}",
        )
        plt.plot(
            self.epochs[:-skip_last] if skip_last > 0 else self.epochs,
            valid_vals[:-skip_last] if skip_last > 0 else valid_vals,
            label=f"Valid {metric_name}",
        )
        plt.xlabel("Epochs")
        plt.ylabel(metric_name)
        plt.title(f"{metric_name} vs. Epochs")
        plt.legend()
        if save_path:
            plt.savefig(save_path)
        plt.show()
        return plt

    def plot_all_metrics(self, skip_last=0, save_path=None):
        """
        Plots all available metrics in subplots.

        Args:
            skip_last (int): Number of last points to skip for all metrics
        """
        num_metrics = len(self.metrics)
        fig, axes = plt.subplots(num_metrics, 1, figsize=(10, 6 * num_metrics))

        # If there's only one metric, axes is not a list, so make sure we handle that.
        if num_metrics == 1:
            axes = [axes]

        for i, (metric_name, metric_data) in enumerate(self.metrics.items()):
            train_vals = [d["train"] for d in metric_data]
            valid_vals = [d["valid"] for d in metric_data]

            # convert to cpu numpy if necessary
            train_vals = [
                val.item() if isinstance(val, torch.Tensor) else val
                for val in train_vals
            ]
            valid_vals = [
                val.item() if isinstance(val, torch.Tensor) else val
                for val in valid_vals
            ]

            # Plot the data
            axes[i].plot(
                self.epochs[:-skip_last] if skip_last > 0 else self.epochs,
                train_vals[:-skip_last] if skip_last > 0 else train_vals,
                label=f"Train {metric_name}",
            )
            axes[i].plot(
                self.epochs[:-skip_last] if skip_last > 0 else self.epochs,
                valid_vals[:-skip_last] if skip_last > 0 else valid_vals,
                label=f"Valid {metric_name}",
            )
            axes[i].set_xlabel("Epochs")
            axes[i].set_ylabel(metric_name)
            axes[i].set_title(f"{metric_name} vs. Epochs")
            axes[i].legend()

        # Adjust layout
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()
        return plt

    def get_metrics_history(self):
        """
        Returns a dictionary containing the history of all recorded metrics.

        Returns:
            dict: A dictionary with metric names as keys and lists of values as values
        """
        return self.metrics


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
