import torch
import pytest
from unittest.mock import Mock
from openml_pytorch.trainer import DefaultConfigGenerator

@pytest.fixture
def generate_default_config():
    return DefaultConfigGenerator()

@pytest.mark.parametrize("attr_model", [
    "device",
    "loss_fn",
    "predict",
    "predict_proba",
    "epoch_count",
    "verbose",
    "opt_kwargs",
    "scheduler_kwargs",
])
def test_model_config_fields_not_none(generate_default_config, attr_model):
    model_config = generate_default_config.return_model_config()
    assert getattr(model_config, attr_model) is not None, f"{attr_model} should not be None"

def test_model_config_fields_scheduler(generate_default_config):
    model_config = generate_default_config.return_model_config()
    assert model_config.scheduler == None


@pytest.mark.parametrize("attr_data", [
    "type_of_data",
    "perform_validation",
    "sanitize",
    "retype_labels",
    "image_size",
    "batch_size",
    "validation_split",
    "transform",
    "transform_test"
])
def test_data_config_fields_not_none(generate_default_config, attr_data):
    data_config = generate_default_config.return_data_config()
    assert getattr(data_config, attr_data) is not None, f"{attr_data} should not be None"

def test_data_config_fields_augmentation(generate_default_config):
    data_config = generate_default_config.return_data_config()
    assert data_config.data_augmentation == None
