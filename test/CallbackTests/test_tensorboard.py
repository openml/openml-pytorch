import pytest
import torch
from torch.utils.tensorboard.writer import SummaryWriter
import pathlib
import os
import shutil

from openml_pytorch.callbacks.tensorboard import TensorBoardCallback

@pytest.fixture
def tensorboardcallback():
    tensorboard_writer = SummaryWriter(
        comment="expname",
        log_dir=f"tensorboard_logs/test/testingdir",
    )
    tb = TensorBoardCallback(writer= tensorboard_writer)
    tb.writer = tensorboard_writer
    tb.model = torch.nn.Sequential()
    tb.xb = torch.tensor(1. )
    return tb

def test_begin_batch_saved_graph(tensorboardcallback):
    tensorboardcallback.begin_batch()
    assert tensorboardcallback.writer is not None
    assert tensorboardcallback.saved_graph is not None

def test_after_fit_writer(tensorboardcallback):
    tensorboardcallback.begin_batch()
    tensorboardcallback.after_fit()
    assert tensorboardcallback.writer.log_dir == 'tensorboard_logs/test/testingdir'
    assert tensorboardcallback.writer.default_bins is not None
    assert "events.out.tfevents" in os.listdir("tensorboard_logs/test/testingdir")[0] 
    shutil.rmtree(pathlib.Path("tensorboard_logs/test"))
    

    
