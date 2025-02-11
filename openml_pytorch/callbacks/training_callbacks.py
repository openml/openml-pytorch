from .callback import Callback


class TrainEvalCallback(Callback):
    """
    TrainEvalCallback class is a custom callback used during the training
    and validation phases of a machine learning model to perform specific
    actions at the beginning and after certain events.

    Methods:

    begin_fit():
        Initialize the number of epochs and iteration counts at the start
        of the fitting process.

    after_batch():
        Update the epoch and iteration counts after each batch during
        training.

    begin_epoch():
        Set the current epoch, switch the model to training mode, and
        indicate that the model is in training.

    begin_validate():
        Switch the model to evaluation mode and indicate that the model
        is in validation.
    """

    def begin_fit(self):
        self.run.n_epochs = 0
        self.run.n_iter = 0

    def after_batch(self):
        if not self.in_train:
            return
        self.run.n_epochs += 1.0 / self.iters
        self.run.n_iter += 1

    def begin_epoch(self):
        self.run.n_epochs = self.epoch
        self.model.train()
        self.run.in_train = True

    def begin_validate(self):
        self.model.eval()
        self.run.in_train = False


class CancelTrainException(Exception):
    pass


class CancelEpochException(Exception):
    pass


class CancelBatchException(Exception):
    pass


class TestCallback(Callback):
    """
    TestCallback class is a custom callback used to test the training loop by stopping the training process after 2 iterations. Useful for debugging and testing purposes, not intended for actual training.
    """

    def after_step(self):
        if self.n_iter >= 1:
            raise CancelTrainException()
