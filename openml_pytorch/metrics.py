import torch
import numpy as np


def accuracy(out, yb):
    """
    Calculates the accuracy of model predictions.

    Parameters:
    out: A tensor containing the model's predicted outputs.
    yb: A tensor containing the actual labels.

    Returns:
    The proportion of correct predictions as a float.
    """
    return (torch.argmax(out, dim=1) == yb.long()).float().mean()

def accuracy_topk(out, yb, k=5):
    """

    Computes the top-k accuracy of the given model outputs.

    Args:
        out (torch.Tensor): The output predictions of the model, of shape (batch_size, num_classes).
        yb (torch.Tensor): The ground truth labels, of shape (batch_size,).
        k (int, optional): The number of top predictions to consider. Default is 5.

    Returns:
        float: The top-k accuracy as a float value.

    The function calculates how often the true label is among the top-k predicted labels.
    """
    return (torch.topk(out, k, dim=1)[1] == yb.long().unsqueeze(1)).float().mean()

def sigmoid(x):
    """
    Computes the sigmoid function

    The sigmoid function is defined as 1 / (1 + exp(-x)). This function is used
    to map any real-valued number into the range (0, 1). It is widely used in
    machine learning, especially in logistic regression and neural networks.

    Args:
        x (numpy.ndarray or float): The input value or array over which the
        sigmoid function should be applied.

    Returns:
        numpy.ndarray or float: The sigmoid of the input value or array.
    """
    return 1 / (1 + np.exp(-x))
