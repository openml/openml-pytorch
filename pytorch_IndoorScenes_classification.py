import torch
import torch.nn as nn
import torch.nn.functional as F
import openml_pytorch
import openml
import warnings

# warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore')


def evaluate_torch_model(model):
    # Download CV splits
    task = openml.tasks.get_task(362070)
    # Evaluate model
    run = openml.runs.run_model_on_task(model, task, avoid_duplicate_runs=False)
    # Publish
    run = openml_pytorch.add_onnx_to_run(run) # Optional, to inspect afterward
    run.publish()
    return run

from torchvision import models
from torchvision.transforms import v2

class Model2(nn.Module):
    def __init__(self, num_classes=67):
        super(Model2, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(13456, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes) 

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Training parameters
openml_pytorch.config.batch_size = 32
openml_pytorch.config.epoch_count = 1
openml_pytorch.config.image_size = 128

transforms = v2.Compose([
    v2.RandomResizedCrop(size=(224, 224), antialias=True),
    v2.RandomHorizontalFlip(p=0.5),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

openml_pytorch.config.data_augemntation = transforms 
# openml_pytorch.config.perform_validation = True

openml.config.apikey = 'key'
openml_pytorch.config.file_dir = openml.config.get_cache_directory()+'/datasets/45923/Images/'
openml_pytorch.config.filename_col = "Filename"

# Run
run = evaluate_torch_model(Model2()) # Replace with your model
print('URL for run: %s/run/%d?api_key=%s' % (openml.config.server, run.run_id, openml.config.apikey)) 