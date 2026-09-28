import math

import pytest
import torch
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import roc_auc_score

import openml_pytorch.metrics as metrics


def test_roc_auc_matches_sklearn_for_binary_logits():
    logits = torch.tensor(
        [[3.0, -1.0], [0.0, 2.0], [0.0, 0.5], [2.0, 1.0], [0.0, 1.0]],
        dtype=torch.float64,
    )
    labels = torch.tensor([0, 1, 0, 1, 0])
    expected = roc_auc_score(labels.numpy(), (logits[:, 1] - logits[:, 0]).numpy())

    assert metrics.roc_auc(logits, labels) == pytest.approx(expected, rel=0, abs=1e-9)


@pytest.mark.parametrize("dtype", [torch.float64, torch.bfloat16])
def test_roc_auc_matches_sklearn_for_multiclass_logits(dtype):
    logits = torch.tensor(
        [
            [3.0, 0.1, -1.0],
            [0.2, 2.0, 0.1],
            [0.1, 0.2, 2.0],
            [1.0, 2.2, 0.0],
            [0.1, 1.0, 2.2],
            [2.1, 0.0, 1.0],
            [1.5, 0.6, 0.7],
            [0.8, 1.1, 0.2],
            [0.9, 0.2, 0.1],
        ],
        dtype=dtype,
    )
    labels = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 0])
    probabilities = torch.softmax(logits.double(), dim=1).numpy()
    for average in ("macro", "weighted"):
        expected = roc_auc_score(
            labels.numpy(),
            probabilities,
            multi_class="ovr",
            average=average,
            labels=[0, 1, 2],
        )
        assert metrics.roc_auc(logits, labels, average=average) == pytest.approx(
            expected, rel=0, abs=1e-9
        )
        if average == "macro":
            assert metrics.roc_auc(logits, labels) == pytest.approx(
                expected, rel=0, abs=1e-9
            )


def test_roc_auc_returns_nan_when_multiclass_label_is_missing():
    logits = torch.tensor([[3.0, 0.0, 0.0], [0.0, 3.0, 0.0]] * 2)
    labels = torch.tensor([0, 1, 0, 1])

    with pytest.warns(UndefinedMetricWarning):
        assert math.isnan(metrics.roc_auc(logits, labels))
