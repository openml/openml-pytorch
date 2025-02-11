from .callback import Callback

class TensorBoardCallback(Callback):
    """
    Log specific things to TensorBoard.
    - Model
    """

    def __init__(self, writer):
        self.writer = writer

    def begin_batch(self):
        if "saved_graph" not in self.__dict__ or not self.saved_graph:
            self.writer.add_graph(self.model, self.xb)
            self.saved_graph = True

    def after_fit(self):
        # check if tensorboard writer is available
        try:
            # add loss and learning rate  to tensorboard
            self.writer.add_scalar("Loss", self.run.loss, self.n_iter)
            self.writer.add_scalar(
                "Learning rate", self.run.opt.param_groups[0]["lr"], self.n_iter
            )
        except Exception as e:
            print(f"Error: {e}")
        self.writer.close()
