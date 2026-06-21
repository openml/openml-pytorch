import pytest
import os
import shutil
import pathlib

from openml_pytorch.callbacks.csv_logger import CSVLoggerCallback

class MockRun:
    def __init__(self):
        self.cbs = []
        self.epoch = 0

@pytest.fixture
def csv_logger_callback():
    log_dir = "csv_logs/test/testingdir"
    exp_name = "test_exp"
    cb = CSVLoggerCallback(log_dir=log_dir, experiment_name=exp_name)
    cb.run = MockRun()
    return cb

def test_begin_fit(csv_logger_callback):
    csv_logger_callback.begin_fit()
    assert csv_logger_callback.file is not None
    assert csv_logger_callback.writer is not None
    assert os.path.exists(csv_logger_callback.log_path)
    csv_logger_callback.after_fit()

def test_after_fit(csv_logger_callback):
    csv_logger_callback.begin_fit()
    csv_logger_callback.after_fit()
    assert csv_logger_callback.file.closed
    
    # cleanup
    shutil.rmtree(pathlib.Path("csv_logs/test"))
