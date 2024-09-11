import torch
import numpy as np


def accuracy(out, yb):
    return (torch.argmax(out, dim=1) == yb.long()).float().mean()

def accuracy_topk(out, yb, k=5):
    return (torch.topk(out, k, dim=1)[1] == yb.long().unsqueeze(1)).float().mean()

def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
