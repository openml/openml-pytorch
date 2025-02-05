#!/usr/bin/env python
# coding: utf-8

# # Basic Image classification task

# In[1]:


# openml imports
import openml
import openml_pytorch
from openml_pytorch.callbacks import TestCallback
from openml_pytorch.metrics import accuracy
from openml_pytorch.trainer import OpenMLDataModule, OpenMLTrainerModule, convert_to_rgb

# pytorch imports
from torchvision.transforms import Compose, Resize, ToPILImage, ToTensor, Lambda
import torchvision

# other imports
import logging
import warnings

# set up logging
openml.config.logger.setLevel(logging.DEBUG)
openml_pytorch.config.logger.setLevel(logging.DEBUG)
# openml.config.start_using_configuration_for_example()
warnings.simplefilter(action="ignore")


# ## Data

# ### Define image transformations
#

# In[2]:


transform = Compose(
    [
        ToPILImage(),  # Convert tensor to PIL Image to ensure PIL Image operations can be applied.
        Lambda(convert_to_rgb),  # Convert PIL Image to RGB if it's not already.
        Resize((64, 64)),  # Resize the image.
        ToTensor(),  # Convert the PIL Image back to a tensor.
    ]
)


# ### Configure the Data Module and Choose a Task
# - Make sure the data is present in the `file_dir` directory, and the `filename_col` is correctly set along with this column correctly pointing to where your data is stored.
#

# In[3]:


data_module = OpenMLDataModule(
    type_of_data="image",
    file_dir="datasets",
    filename_col="image_path",
    target_mode="categorical",
    target_column="label",
    batch_size=64,
    transform=transform,
)

# Download the OpenML task for tiniest imagenet
task = openml.tasks.get_task(362128)


# ## Model

# In[4]:


model = torchvision.models.resnet18(num_classes=200)


# ## Train your model on the data
# - Note that by default, OpenML runs a 10 fold cross validation on the data. You cannot change this for now.

# In[5]:


trainer = OpenMLTrainerModule(
    experiment_name="Tiny ImageNet, 1 epoch",
    data_module=data_module,
    verbose=True,
    epoch_count=1,
    metrics=[accuracy],
    # remove the TestCallback when you are done testing your pipeline. Having it here will make the pipeline run for a very short time.
    callbacks=[
        TestCallback,
    ],
)
openml_pytorch.config.trainer = trainer
run = openml.runs.run_model_on_task(model, task, avoid_duplicate_runs=False)
print(run)
# ## testing push

# In[6]:


run.publish()
