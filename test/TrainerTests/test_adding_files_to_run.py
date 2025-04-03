import pytest
import json
import torch
from unittest.mock import MagicMock, patch
import openml
from pathlib import Path
from openml_pytorch.run_utils import (
    add_file_to_run,
    safe_add,
)
from pathlib import Path
import pytest
import openml
import openml_pytorch as op
import torchvision
import torch
from openml_pytorch.metrics import accuracy
from openml_pytorch.trainer import convert_to_rgb
from torchvision.transforms import Compose, Resize, ToPILImage, ToTensor, Lambda

run = openml.runs.get_run(10595300)
@pytest.fixture
def setup_run():
    global run
    run = openml.runs.get_run(10595300)
    return run

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

def test_add_file_to_run():
    global run   
    run._get_file_elements = MagicMock(return_value={})
    file_content = "sample content"
    run = add_file_to_run(run, file_content, "test_file")
    assert "test_file" in run._get_file_elements()

def test_add_file_to_run_raises_type_error():
    global run   
    with pytest.raises(TypeError):
        add_file_to_run(run, Path("/some/path"), "test_file")

def test_safe_add(setup_trainer):
    trainer = setup_trainer
    trainer.data_module = "value"
    attribute_dict = {}
    safe_add(attribute_dict, trainer, "data_module", "key")
    assert attribute_dict["key"] == "value"
