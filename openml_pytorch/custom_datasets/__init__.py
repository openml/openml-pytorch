"""
This module contains the custom datasets for OpenML datasets.
"""
from .image_dataset import OpenMLImageDataset
from .tabular_dataset import OpenMLTabularDataset
from .generic_dataset import GenericDataset

__all__ = ["OpenMLTabularDataset", "OpenMLImageDataset", "GenericDataset"]