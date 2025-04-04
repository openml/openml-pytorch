"""
This module provides utility functions for evaluating model performance and activation functions.
It includes functions to compute the accuracy, top-k accuracy of model predictions, and the sigmoid function.
"""

import torch


def accuracy(out, yb):
    """

    Computes the accuracy of model predictions.

    Parameters:
    out (Tensor): The output tensor from the model, containing predicted class scores.
    yb (Tensor): The ground truth labels tensor.

    Returns:
    Tensor: The mean accuracy of the predictions, computed as a float tensor.
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


def f1_score(out, yb):
    """
    Computes the F1 score for the given model outputs and true labels.

    Args:
        out (torch.Tensor): The output predictions of the model, of shape (batch_size, num_classes).
        yb (torch.Tensor): The ground truth labels, of shape (batch_size,).

    Returns:
        float: The F1 score as a float value.
    """
    pred = torch.argmax(out, dim=1)
    tp = torch.sum((pred == 1) & (yb == 1)).float()
    fp = torch.sum((pred == 1) & (yb == 0)).float()
    fn = torch.sum((pred == 0) & (yb == 1)).float()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)

    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

    return f1
