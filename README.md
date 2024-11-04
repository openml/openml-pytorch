# Pytorch extension for OpenML python

Pytorch extension for [openml-python API](https://github.com/openml/openml-python). This library provides a simple way to run your Pytorch models on OpenML tasks. 

For a more native experience, PyTorch itself provides OpenML integrations for some tasks. You can find more information [here](<Integrations of OpenML in PyTorch.md>).

#### Installation Instructions:

`pip install openml-pytorch`

PyPi link https://pypi.org/project/openml-pytorch/

Set the API key for OpenML from the command line:
```bash
openml configure apikey <your API key>
```

#### Usage
Import openML libraries
```python
import torch.nn
import torch.optim

import openml_pytorch.config
import openml
import logging

from openml_pytorch.trainer import OpenMLTrainerModule
from openml_pytorch.trainer import OpenMLDataModule
from torchvision.transforms import Compose, Resize, ToPILImage, ToTensor, Lambda
import torchvision
from openml_pytorch.trainer import convert_to_rgb

```
Create a pytorch model and get a task from openML
```python
model = torchvision.models.efficientnet_b0(num_classes=200)
# Download the OpenML task for tiniest imagenet
task = openml.tasks.get_task(362128)
```
Download the task from openML and define Data and Trainer configuration
```python
transform = Compose(
    [
        ToPILImage(),  # Convert tensor to PIL Image to ensure PIL Image operations can be applied.
        Lambda(
            convert_to_rgb
        ),  # Convert PIL Image to RGB if it's not already.
        Resize(
            (64, 64)
        ),  # Resize the image.
        ToTensor(),  # Convert the PIL Image back to a tensor.
    ]
)
data_module = OpenMLDataModule(
    type_of_data="image",
    file_dir="datasets",
    filename_col="image_path",
    target_mode="categorical",
    target_column="label",
    batch_size = 64,
    transform=transform
)
trainer = OpenMLTrainerModule(
    data_module=data_module,
    verbose = True,
    epoch_count = 1,
)
openml_pytorch.config.trainer = trainer
```
Run the model on the task
```python
run = openml.runs.run_model_on_task(model, task, avoid_duplicate_runs=False)
run.publish()
print('URL for run: %s/run/%d' % (openml.config.server, run.run_id))
```
Note: The input layer of the network should be compatible with OpenML data output shape. Please check [examples](/examples/) for more information.

Additionally, if you want to publish the run with onnx file, then you must call ```openml_pytorch.add_onnx_to_run()``` immediately before ```run.publish()```. 

```python
run = openml_pytorch.add_onnx_to_run(run)
```
