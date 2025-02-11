from .callback import Callback
class PutDataOnDeviceCallback(Callback):
    """
    PutDataOnDevice class is a custom callback used to move the input data and target labels to the device (CPU or GPU) before passing them to the model.

    Methods:
        begin_fit: Moves the model to the device at the beginning of the fitting process.
        begin_batch: Moves the input data and target labels to the device at the beginning of each batch.
    """

    def __init__(self, device):
        self.device = device

    def begin_fit(self):
        self.model.to(self.device)

    def begin_batch(self):
        self.run.xb, self.run.yb = self.xb.to(self.device), self.yb.to(self.device)

    def after_pred(self):
        self.run.pred = self.run.pred.to(self.device)
        self.run.yb = self.run.yb.to(self.device)


