from PIL import Image
import numpy as np
from openml_pytorch.trainer import convert_to_rgb


def test_convert_to_rgb_already_rgb():
    image = Image.new("RGB", (64, 64))
    result = convert_to_rgb(image)
    assert result.mode == "RGB"


def test_convert_to_rgb_grayscale():
    image = Image.new("L", (64, 64))  # Grayscale
    result = convert_to_rgb(image)
    assert result.mode == "RGB"
