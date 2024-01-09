"""
PyTorch sequential regression model example
==================

An example of a sequential network that solves a regression task used as an OpenML flow.
"""

import torch.nn
import torch.optim

import openml
import openml_pytorch

import logging

############################################################################
# Enable logging in order to observe the progress while running the example.
openml.config.logger.setLevel(logging.DEBUG)
openml_pytorch.config.logger.setLevel(logging.DEBUG)
############################################################################
openml.config.apikey = 'd7f058387fb3c8ba41e1ae61ebd999a0'
############################################################################
# Define a sequential network with 1 input layer, 3 hidden layers and 1 output
# layer, using the LeakyReLU activation function and a dropout rate of 0.5.
model = torch.nn.Sequential(
    torch.nn.Linear(in_features=13, out_features=256),
    torch.nn.LeakyReLU(),
    torch.nn.Dropout(),
    torch.nn.Linear(in_features=256, out_features=256),
    torch.nn.LeakyReLU(),
    torch.nn.Dropout(),
    torch.nn.Linear(in_features=256, out_features=256),
    torch.nn.LeakyReLU(),
    torch.nn.Dropout(),
    torch.nn.Linear(in_features=256, out_features=256),
    torch.nn.LeakyReLU(),
    torch.nn.Dropout(),
    torch.nn.Linear(in_features=256, out_features=1)
)
############################################################################

############################################################################
# Download the OpenML task for the cholesterol dataset.
task = openml.tasks.get_task(2295)
############################################################################
# Run the model on the task (requires an API key).
run = openml.runs.run_model_on_task(model, task, avoid_duplicate_runs=False)
# Publish the experiment on OpenML (optional, requires an API key).
run.publish()

print('URL for run: %s/run/%d' % (openml.config.server, run.run_id))

############################################################################
