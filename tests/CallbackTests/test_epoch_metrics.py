import math
from functools import partial
from types import SimpleNamespace

import pytest
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from openml_pytorch.callbacks.recording import AvgStatsCallback, Recorder
from openml_pytorch.metrics import accuracy, accuracy_topk
from openml_pytorch.trainer import Learner, ModelRunner


def run_one_epoch(features, labels, class_count, metrics, batch_size=64):
    model = torch.nn.Linear(features.shape[1], class_count)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loader = DataLoader(TensorDataset(features, labels), batch_size=batch_size)
    data = SimpleNamespace(train_dl=loader, valid_dl=loader)
    learner = Learner(
        model, optimizer, torch.nn.CrossEntropyLoss(), None, data, list(range(class_count))
    )
    runner = ModelRunner(cb_funcs=[Recorder, partial(AvgStatsCallback, metrics)])
    runner.fit(1, learner)
    return model, runner.cbs[1]


def test_auroc_uses_whole_validation_epoch():
    torch.manual_seed(0)
    features = torch.randn(2560, 8)
    labels = torch.zeros(2560, dtype=torch.long)
    labels[torch.randperm(2560)[:51]] = 1

    def epoch_auroc(outputs, targets):
        probabilities = torch.softmax(outputs.detach().cpu(), dim=1)[:, 1]
        return roc_auc_score(targets.cpu().numpy(), probabilities.numpy())

    model, recorder = run_one_epoch(features, labels, 2, [epoch_auroc])
    recorded_auroc = recorder.metrics["epoch_auroc"][0]["valid"]
    with torch.no_grad():
        expected_auroc = epoch_auroc(model(features), labels)

    assert math.isfinite(recorded_auroc)
    assert recorded_auroc == pytest.approx(expected_auroc, abs=1e-6)


def test_partial_metric_name_appears_in_recording_and_output(capsys):
    torch.manual_seed(1)
    features = torch.randn(34, 8)
    labels = torch.arange(34) % 4

    model, recorder = run_one_epoch(
        features,
        labels,
        4,
        [partial(accuracy_topk, k=2), partial(accuracy_topk, k=3)],
        batch_size=8,
    )

    output_lines = capsys.readouterr().out.splitlines()
    train_line = next(line for line in output_lines if line.startswith("Train:"))
    valid_line = next(line for line in output_lines if line.startswith("Valid:"))
    with torch.no_grad():
        predictions = model(features)
    for k in (2, 3):
        name = f"accuracy_topk_k={k}"
        assert name in recorder.metrics
        assert name in train_line
        assert name in valid_line
        assert float(recorder.metrics[name][0]["valid"]) == pytest.approx(
            float(accuracy_topk(predictions, labels, k=k))
        )


def test_accuracy_matches_whole_epoch_and_metrics_run_once_per_phase():
    torch.manual_seed(2)
    features = torch.randn(96, 8)
    labels = torch.arange(96) % 4
    observed_shapes = []

    def observed_accuracy(outputs, targets):
        observed_shapes.append((tuple(outputs.shape), tuple(targets.shape)))
        return accuracy(outputs, targets)

    model, recorder = run_one_epoch(
        features, labels, 4, [accuracy, observed_accuracy], batch_size=16
    )
    with torch.no_grad():
        predicted_labels = model(features).argmax(dim=1)
    expected_accuracy = accuracy_score(labels.numpy(), predicted_labels.numpy())

    recorded_accuracy = float(recorder.metrics["accuracy"][0]["valid"])
    assert recorded_accuracy == pytest.approx(expected_accuracy)
    assert observed_shapes == [((96, 4), (96,)), ((96, 4), (96,))]
