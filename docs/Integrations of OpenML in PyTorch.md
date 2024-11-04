# Integrations of OpenML in PyTorch

Along with this PyTorch API, OpenML is also integrated in PyTorch through the following modules.

## Reinforcement Learning
- The RL library [TorchRL](https://pytorch.org/rl/stable/reference/envs.html) supports loading OpenML datasets as part of inbuilt modules.  

### TorchRL - [OpenMLExperienceReplay](https://pytorch.org/rl/main/reference/generated/torchrl.data.datasets.OpenMLExperienceReplay.html)
- Experience replay is a technique used in reinforcement learning to improve the stability and performance of deep reinforcement learning algorithms by storing and reusing experience tuples.
- This module provides a direct interface to OpenML datasets to be used in experience replay buffers.

```python
exp = OpenMLExperienceReplay("adult_onehot", batch_size=2)
# the following datasets are supported: "adult_num", "adult_onehot", "mushroom_num", "mushroom_onehot", "covertype", "shuttle" and "magic"
print(exp.sample())
```

### TorchRL -   [OpenMLEnv](https://pytorch.org/rl/stable/_modules/torchrl/envs/libs/openml.html#OpenMLEnv)
- Bandits are a class of RL problems where the agent has to choose between multiple actions and receives a reward based on the action chosen.
- This module provides an environment interface to OpenML data to be used in bandits contexts.
- Given a dataset name (obtained from [openml datasets](https://www.openml.org/search?type=data)), it returns a PyTorch environment that can be used in PyTorch training loops.

```python
env = OpenMLEnv("adult_onehot", batch_size=[2, 3])
# the following datasets are supported: "adult_num", "adult_onehot", "mushroom_num", "mushroom_onehot", "covertype", "shuttle" and "magic"
print(env.reset())
```