"""Tests for Deep Learning Capstone. Run: pytest tests/ -v"""
import torch
import pytest


def test_resblock_forward():
    from train import ResBlock
    block = ResBlock(64)
    x = torch.randn(2, 64, 16, 16)
    out = block(x)
    assert out.shape == x.shape


def test_cifar10net_forward():
    from train import CIFAR10Net
    model = CIFAR10Net(use_skip=True)
    x = torch.randn(4, 3, 32, 32)
    out = model(x)
    assert out.shape == (4, 10)


def test_cifar10net_no_skip():
    from train import CIFAR10Net
    model = CIFAR10Net(use_skip=False)
    x = torch.randn(4, 3, 32, 32)
    out = model(x)
    assert out.shape == (4, 10)


def test_gradient_monitor():
    from train import CIFAR10Net
    from diagnostics import GradientMonitor
    model = CIFAR10Net()
    monitor = GradientMonitor(model)
    x = torch.randn(2, 3, 32, 32)
    loss = model(x).sum()
    loss.backward()
    monitor.record()
    assert len(monitor.grad_history) > 0


def test_training_tracker():
    from diagnostics import TrainingTracker
    tracker = TrainingTracker()
    for i in range(10):
        tracker.log(train_loss=1.0 - i * 0.05, val_loss=1.0 - i * 0.04,
                     train_acc=50 + i * 4, val_acc=50 + i * 3)
    tracker.diagnose()


def test_log_experiment(tmp_path):
    from diagnostics import log_experiment
    import csv
    filepath = str(tmp_path / "test_log.csv")
    config = {"lr": 0.1, "weight_decay": 5e-4, "dropout": 0.1, "augmentation": "crop+flip"}
    log_experiment(filepath, "test_exp", config, 0.95, 0.90, "test note")
    with open(filepath) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert len(rows) == 1
    assert rows[0]["experiment"] == "test_exp"
