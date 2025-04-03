from pathlib import Path
import pytest
import openml
import openml_pytorch as op
import torchvision
import torch
from openml_pytorch.metrics import accuracy
from openml_pytorch.trainer import convert_to_rgb
from torchvision.transforms import Compose, Resize, ToPILImage, ToTensor, Lambda

@pytest.fixture
def setup_data_module():
    transform = Compose(
        [
            ToPILImage(),
            Lambda(convert_to_rgb),
            Resize((64, 64)),
            ToTensor(),
        ]
    )
    BRD = openml.datasets.get_dataset(46770, download_all_files=True)
    data_dir = Path(openml.config.get_cache_directory())/'datasets'/str(BRD.dataset_id)/"BRD_Extended"/"images"
    
    data_module = op.trainer.OpenMLDataModule(
        type_of_data="image",
        file_dir=str(data_dir),
        filename_col="file_path",
        target_mode="categorical",
        target_column="CATEGORY",
        batch_size=32,
        transform=transform
    )
    return data_module

@pytest.fixture
def setup_model():
    return torchvision.models.resnet18(num_classes=315)

@pytest.fixture
def setup_trainer(setup_data_module):
    trainer = op.OpenMLTrainerModule(
        experiment_name="Birds",
        data_module=setup_data_module,
        verbose=True,
        epoch_count=1,  # Use 1 epoch for testing
        metrics=[accuracy],
        callbacks=[op.callbacks.TestCallback],
        opt=torch.optim.Adam,
    )
    op.config.trainer = trainer
    return trainer

@pytest.fixture
def setup_task():
    return openml.tasks.get_task(363465)

def test_data_loading(setup_data_module):
    assert setup_data_module is not None

def test_model_initialization(setup_model):
    assert setup_model is not None
    assert isinstance(setup_model, torchvision.models.ResNet)

def test_training_pipeline(setup_model, setup_task, setup_trainer):
    run = openml.runs.run_model_on_task(setup_model, setup_task, avoid_duplicate_runs=False)
    assert run is not None
    assert setup_trainer.stats.metrics is not None
    assert setup_trainer.plot_all_metrics() is not None
    assert setup_trainer.plot_loss() is not None
    assert setup_trainer.plot_lr() is not None