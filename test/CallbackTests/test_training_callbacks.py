import pytest
from unittest.mock import MagicMock

# Assuming the callbacks are in a module named `callbacks`
from openml_pytorch.callbacks.training_callbacks import (
    TrainEvalCallback,
    TestCallback,
    CancelTrainException,
)


class DummyModel:
    def __init__(self):
        self.training = True

    def train(self):
        self.training = True

    def eval(self):
        self.training = False


class DummyRun:
    def __init__(self):
        self.n_epochs = 0
        self.n_iter = 0
        self.epoch = 5
        self.in_train = True


@pytest.fixture
def train_eval_cb():
    cb = TrainEvalCallback()
    cb.run = DummyRun()
    cb.model = DummyModel()
    cb.iters = 10
    cb.epoch = 5
    cb.in_train = True
    return cb


def test_begin_fit(train_eval_cb):
    train_eval_cb.begin_fit()
    assert train_eval_cb.run.n_epochs == 0
    assert train_eval_cb.run.n_iter == 0


def test_after_batch_in_train(train_eval_cb):
    train_eval_cb.in_train = True
    train_eval_cb.after_batch()
    assert train_eval_cb.run.n_epochs == 0.1
    assert train_eval_cb.run.n_iter == 1


def test_after_batch_not_in_train(train_eval_cb):
    train_eval_cb.in_train = False
    initial_epochs = train_eval_cb.run.n_epochs
    initial_iters = train_eval_cb.run.n_iter
    train_eval_cb.after_batch()
    assert train_eval_cb.run.n_epochs == initial_epochs
    assert train_eval_cb.run.n_iter == initial_iters


def test_begin_epoch(train_eval_cb):
    train_eval_cb.begin_epoch()
    assert train_eval_cb.run.n_epochs == train_eval_cb.epoch
    assert train_eval_cb.model.training is True
    assert train_eval_cb.run.in_train is True


def test_begin_validate(train_eval_cb):
    train_eval_cb.begin_validate()
    assert train_eval_cb.model.training is False
    assert train_eval_cb.run.in_train is False


def test_test_callback_stops_after_step():
    cb = TestCallback()
    cb.n_iter = 1
    with pytest.raises(CancelTrainException):
        cb.after_step()

    cb.n_iter = 0
    try:
        cb.after_step()  # Should not raise
    except CancelTrainException:
        pytest.fail("CancelTrainException raised unexpectedly")
