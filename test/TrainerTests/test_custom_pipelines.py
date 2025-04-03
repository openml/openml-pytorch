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
            Resize((16, 16)),
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
        batch_size=64,
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

def test_custom_scheduler(setup_trainer):
    """
    Test the custom learning rate scheduler.
    """
    # Check if the scheduler is set correctly
    setup_trainer.lr_scheduler = torch.optim.lr_scheduler.StepLR
    setup_trainer.lr_scheduler_kwargs = {"step_size": 1, "gamma": 0.1}
    assert setup_trainer.lr_scheduler == torch.optim.lr_scheduler.StepLR
    assert setup_trainer.lr_scheduler_kwargs == {"step_size": 1, "gamma": 0.1}

def test_custom_optimizer(setup_trainer):
    """
    Test the custom optimizer.
    """
    # Check if the optimizer is set correctly
    setup_trainer.opt = torch.optim.AdamW
    setup_trainer.opt_kwargs = {"lr": 1e-3, "weight_decay": 1e-4}
    assert setup_trainer.opt == torch.optim.AdamW
    assert setup_trainer.opt_kwargs == {"lr": 1e-3, "weight_decay": 1e-4}

def test_if_custom_callbacks_are_added(setup_trainer):
    setup_trainer.callbacks = [op.callbacks.TestCallback]
    assert setup_trainer.cbfs[-1].__name__ == "TestCallback"

def test_if_model_classes_are_same_as_task(setup_trainer):
    num_classes = len(setup_task.class_labels)
    trainer_labels = len(setup_trainer.model_classes)
    assert num_classes == trainer_labels, f"Task classes {num_classes} do not match model classes {trainer_labels}"