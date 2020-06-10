# Pytorch extension for OpenML python

Pytorch extension for [openml-python API](https://github.com/openml/openml-python).

#### Installation Instructions:

`pip install openml-pytorch`

PyPi link https://pypi.org/project/openml-pytorch/

#### Usage
Import openML libraries
```python
import openml
import openml_pytorch
import openml_pytorch.layers
import openml_pytorch.config

```
Create a torch model
```python
model = torch.nn.Sequential(
    processing_net,
    features_net,
    results_net
)
```
Download the task from openML and run the model on task.
```python
task = openml.tasks.get_task(3573)
run = openml.runs.run_model_on_task(model, task, avoid_duplicate_runs=False)
run.publish()
print('URL for run: %s/run/%d' % (openml.config.server, run.run_id))
```
Note: The input layer of the network should be compatible with OpenML data output shape. Please check examples for more information.