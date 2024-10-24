from .extension import PytorchExtension
from . import config
from . import layers
from . import trainer
from . import data
from openml.extensions import register_extension
import torch
import io
import onnx


__all__ = ['PytorchExtension', 'config', 'layers','add_onnx_to_run', 'trainer']

register_extension(PytorchExtension)

def add_onnx_to_run(run):
    
    run._old_get_file_elements = run._get_file_elements
    
    def modified_get_file_elements():
        elements = run._old_get_file_elements()
        elements["onnx_model"] = ("model.onnx", extension.last_models)
        return elements
    
    run._get_file_elements = modified_get_file_elements
    return run
    