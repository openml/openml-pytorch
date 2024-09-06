"""
PyTorch image classification model example
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

import warnings
import pandas as pd

# Suppress FutureWarning messages
warnings.simplefilter(action='ignore')

############################################################################
# Enable logging in order to observe the progress while running the example.
openml.config.logger.setLevel(logging.DEBUG)
openml_pytorch.config.logger.setLevel(logging.DEBUG)
############################################################################

############################################################################
import torch.nn as nn
import torch.nn.functional as F

# Example model. You can do better :)
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(13456, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 19) # To user - Remember to set correct size of last layer. 

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()

############################################################################
openml.config.apikey = 'key'
openml_pytorch.config.file_dir = openml.config.get_cache_directory()+'/datasets/44312/PNU_Micro/images/'
openml_pytorch.config.filename_col = "FILE_NAME"
openml_pytorch.config.perform_validation = False
#You can set the device type here, 
# alternatively config auto selects it for you depending on the device availability. 
openml_pytorch.config.device = torch.device("cpu") 
############################################################################
# The main network, composed of the above specified networks.
model = net

############################################################################
# Download the OpenML task for the Meta_Album_PNU_Micro dataset.
task = openml.tasks.get_task(361987)

############################################################################
# Run the model on the task (requires an API key).m
run = openml.runs.run_model_on_task(model, task, avoid_duplicate_runs=False)

# If you want to publish the run with the onnx file, 
# then you must call openml_pytorch.add_onnx_to_run() immediately before run.publish(). 
# When you publish, onnx file of last trained model is uploaded. 
# Careful to not call this function when another run_model_on_task is called in between, 
# as during publish later, only the last trained model (from last run_model_on_task call) is uploaded.   
run = openml_pytorch.add_onnx_to_run(run)
run.publish()

print('URL for run: %s/run/%d' % (openml.config.server, run.run_id))
############################################################################

# Visualize model in netron

from urllib.request import urlretrieve

published_run = openml.runs.get_run(run.run_id)
url = 'https://api.openml.org/data/download/{}/model.onnx'.format(published_run.output_files['onnx_model'])

file_path, _ = urlretrieve(url, 'model.onnx')

import netron
# Visualize the ONNX model using Netron
netron.start(file_path)



