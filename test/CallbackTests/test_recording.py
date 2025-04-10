import pytest
import torch
from unittest.mock import MagicMock

from openml_pytorch.callbacks.recording import Recorder, AvgStatsCallback, AvgStats
from openml_pytorch.metrics import accuracy

class DummyOpt:
    def __init__(self, lrs):
        self.param_groups = [
            {"lr":lr} for lr in lrs
        ]

@pytest.fixture
def recorder():
    rec = Recorder()
    rec.opt = DummyOpt([0.01,0.001])
    rec.in_train = True
    rec.loss = torch.tensor(0.5)
    rec.metrics = []
    rec.run = MagicMock()
    rec.run.cbs = [Recorder]
    return rec

def test_begin_fit_initalizes_lists(recorder):
    recorder.begin_fit()
    assert len(recorder.lrs) == 2
    assert recorder.losses == []
    assert recorder.metrics == {}
    assert recorder.epochs == []
    assert recorder.current_epoch == 0


def test_begin_epoch_increments_epochs(recorder):
    recorder.begin_fit()
    recorder.begin_epoch()
    assert recorder.current_epoch == 1

def test_after_batch_records_lr_loss(recorder):
    recorder.begin_fit()
    recorder.after_batch()
    dummy_cb = AvgStatsCallback(metrics=[lambda pred, yb: (pred == yb).float().mean()])
    dummy_cb.train_stats.count = 1

    dummy_cb.train_stats.tot_loss = torch.tensor(1)
    dummy_cb.train_stats.tot_mets = [torch.tensor(1)]
    dummy_cb.train_stats.metrics[0].__name__ = "accuracy"

    dummy_cb.valid_stats.count = 1

    dummy_cb.valid_stats.tot_loss = torch.tensor(1)
    dummy_cb.valid_stats.tot_mets = [torch.tensor(1)]
    dummy_cb.valid_stats.metrics[0].__name__ = "accuracy"
    recorder.run.cbs = [dummy_cb]
    recorder.current_epoch = 1
    recorder.after_epoch()
    assert recorder.epochs == [1]
    assert recorder.metrics["accuracy"][0] == {"train": torch.tensor(1.), "valid": torch.tensor(1.)}


def test_get_metrics_history_returns_dict(recorder):
    recorder.begin_fit()
    recorder.metrics = {"accuracy": [{"train": torch.tensor(1.), "valid": torch.tensor(1.)}]}
    history = recorder.get_metrics_history()
    assert isinstance(history, dict)
    assert "accuracy" in history
    assert history["accuracy"][0]["train"] == torch.tensor(1.)

def test_plot_lr_returns_plot(recorder):
    recorder.begin_fit()
    recorder.lrs[0] = [0.01, 0.009, 0.008]
    plot = recorder.plot_lr(pgid=0)
    assert plot is not None

def test_plot_loss_returns_plot(recorder):
    recorder.begin_fit()
    recorder.losses = [torch.tensor(0.5), torch.tensor(0.4), torch.tensor(0.3)]
    plot = recorder.plot_loss()
    assert plot is not None

