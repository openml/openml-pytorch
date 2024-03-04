"""
PyTorch image classification model using pre-trained ResNet model example
==================

An example of a pytorch network that classifies meta album images.
"""

import torch.nn
import torch.optim

import openml
import openml_pytorch
import openml_pytorch.layers
import openml_pytorch.config
import logging

############################################################################
# Enable logging in order to observe the progress while running the example.
openml.config.logger.setLevel(logging.DEBUG)
openml_pytorch.config.logger.setLevel(logging.DEBUG)
############################################################################

############################################################################
import torch.nn as nn
import torch.nn.functional as F

# Example model. You can do better :)
import torchvision.models as models

# Load the pre-trained ResNet model
model = models.resnet50(pretrained=True)

# Modify the last fully connected layer to the required number of classes
num_classes = 20
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, num_classes)

# Optional: If you're fine-tuning, you may want to freeze the pre-trained layers
for param in model.parameters():
    param.requires_grad = False

# If you want to train the last layer only (the newly added layer)
for param in model.fc.parameters():
    param.requires_grad = True

############################################################################
# Setting an appropriate optimizer 
from openml import OpenMLTask

def custom_optimizer_gen(model: torch.nn.Module, task: OpenMLTask) -> torch.optim.Optimizer:
    return torch.optim.Adam(model.fc.parameters())

openml_pytorch.config.optimizer_gen = custom_optimizer_gen

############################################################################

# openml.config.apikey = 'KEY'
############################################################################
# Download the OpenML task for the Meta_Album_PNU_Micro dataset.
task = openml.tasks.get_task(361152)

############################################################################
# Run the model on the task (requires an API key).m
run = openml.runs.run_model_on_task(model, task, avoid_duplicate_runs=False)

# Publish the experiment on OpenML (optional, requires an API key).
run.publish()

print('URL for run: %s/run/%d' % (openml.config.server, run.run_id))

############################################################################
