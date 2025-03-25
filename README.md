# Pytorch extension for OpenML python

Pytorch extension for [openml-python API](https://github.com/openml/openml-python). This library provides a simple way to run your Pytorch models on OpenML tasks. 

For a more native experience, PyTorch itself provides OpenML integrations for some tasks. You can find more information [here](<Integrations of OpenML in PyTorch.md>).

## Installation Instructions:

<!-- `pip install openml-pytorch` -->
While this project does exist on pypi, while everything is being finalized, it is recommended to install the package directly from the repository. 

```bash
pip install git+https://github.com/openml/openml-pytorch -U
```

PyPi link https://pypi.org/project/openml-pytorch/

Set the API key for OpenML from the command line:
```bash
openml configure apikey <your API key>
```

## Usage
### Load Data from OpenML and Train a Model
```python
# Import libraries
import openml
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Any
from tqdm import tqdm

from openml_pytorch import GenericDataset

# Get dataset by ID and split into train and test
dataset = openml.datasets.get_dataset(20)
X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
X = X.to_numpy(dtype=np.float32)  
y = y.to_numpy(dtype=np.int64)    
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, stratify=y)

# Dataloaders
ds_train = GenericDataset(X_train, y_train)
ds_test = GenericDataset(X_test, y_test)
dataloader_train = torch.utils.data.DataLoader(ds_train, batch_size=64, shuffle=True)
dataloader_test = torch.utils.data.DataLoader(ds_test, batch_size=64, shuffle=False)

# Model Definition
class TabularClassificationModel(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(TabularClassificationModel, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, output_size)
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return x

# Train the model. Feel free to replace this with your own training pipeline. 
trainer = BasicTrainer(
    model = TabularClassificationModel(X_train.shape[1], len(np.unique(y_train))),
    loss_fn = torch.nn.CrossEntropyLoss(),
    opt = torch.optim.Adam,
    dataloader_train = dataloader_train,
    dataloader_test = dataloader_test,
    device= torch.device("mps")
)
trainer.fit(10)
```
## More Complex Image Classification Example

Import openML libraries
```python
# openml imports
import openml
import openml_pytorch as op
from openml_pytorch.callbacks import TestCallback
from openml_pytorch.metrics import accuracy
from openml_pytorch.trainer import convert_to_rgb

# pytorch imports
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision.transforms import Compose, Resize, ToPILImage, ToTensor, Lambda
import torchvision

# other imports
import logging
import warnings

# set up logging
openml.config.logger.setLevel(logging.DEBUG)
op.config.logger.setLevel(logging.DEBUG)
warnings.simplefilter(action='ignore')
## Data
### Define image transformations


transform = Compose(
    [
        ToPILImage(),  # Convert tensor to PIL Image to ensure PIL Image operations can be applied.
        Lambda(convert_to_rgb),  # Convert PIL Image to RGB if it's not already.
        Resize((64, 64)),  # Resize the image.
        ToTensor(),  # Convert the PIL Image back to a tensor.
    ]
)

### Configure the Data Module and Choose a Task
"""
- Make sure the data is present in the `file_dir` directory, and the `filename_col` is correctly set along with this column correctly pointing to where your data is stored. 
"""
data_module = op.OpenMLDataModule(
    type_of_data="image",
    file_dir="datasets",
    filename_col="image_path",
    target_mode="categorical",
    target_column="label",
    batch_size=64,
    transform=transform,
)

# Download the OpenML task for tiniest imagenet
task = openml.tasks.get_task(363295)

## Model
model = torchvision.models.resnet18(num_classes=200)
## Train your model on the data
#- Note that by default, OpenML runs a 10 fold cross validation on the data. You cannot change this for now.
import torch

trainer = op.OpenMLTrainerModule(
    experiment_name= "Tiny ImageNet",
    data_module=data_module,
    verbose=True,
    epoch_count=2,
    metrics= [accuracy],
    # remove the TestCallback when you are done testing your pipeline. Having it here will make the pipeline run for a very short time.
    callbacks=[
        # TestCallback,
    ],
    opt = torch.optim.Adam,
)
op.config.trainer = trainer
run = openml.runs.run_model_on_task(model, task, avoid_duplicate_runs=False)
## View information about your run
### Learning rate and loss plot
trainer.plot_loss()
trainer.plot_lr()
trainer.plot_all_metrics()
### Class labels
trainer.model_classes
## Model Vizualization
#- Sometimes you may want to visualize the model. You can either use netron or tensorboard for this purpose.
### Netron
trainer.export_to_netron()
### Tensorboard
"""
- By default, openml will log the tensorboard logs in the `tensorboard_logs` directory. You can view the logs by running `tensorboard --logdir tensorboard_logs` in the terminal.
"""
## Publish your model to OpenML
"""
- This is Optional, but publishing your model to OpenML will allow you to track your experiments and compare them with others.
- Make sure to set your apikey first.
  - You can find your apikey on your OpenML account page.
"""
trainer.plot_all_metrics()
openml.config.apikey = ''
run = op.add_experiment_info_to_run(run=run, trainer=trainer) 
run.publish()
```