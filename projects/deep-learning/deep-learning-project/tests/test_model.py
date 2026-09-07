"""Tests for Deep Learning Capstone. Run: pytest tests/ -v"""
import torch
import pytest


def test_resblock_forward():
    from src.train import ResBlock
    block = ResBlock(64)
    x = torch.randn(2, 64, 16, 16)
    out = block(x)
    assert out.shape == x.shape


def test_cifar10net_forward():
    from src.train import CIFAR10Net
    model = CIFAR10Net(use_skip=True)
    x = torch.randn(4, 3, 32, 32)
    out = model(x)
    assert out.shape == (4, 10)


def test_cifar10net_no_skip():
    from src.train import CIFAR10Net
    model = CIFAR10Net(use_skip=False)
    x = torch.randn(4, 3, 32, 32)
    out = model(x)
    assert out.shape == (4, 10)


def test_gradient_monitor():
    from src.train import CIFAR10Net
    from src.diagnostics import GradientMonitor
    model = CIFAR10Net()
    monitor = GradientMonitor(model)
    x = torch.randn(2, 3, 32, 32)
    loss = model(x).sum()
    loss.backward()
    monitor.record()
    assert len(monitor.grad_history) > 0


def test_training_tracker():
    from src.diagnostics import TrainingTracker
    tracker = TrainingTracker()
    for i in range(10):
        tracker.log(train_loss=1.0 - i * 0.05, val_loss=1.0 - i * 0.04,
                     train_acc=50 + i * 4, val_acc=50 + i * 3)
    tracker.diagnose()


def test_log_experiment(tmp_path):
    from src.diagnostics import log_experiment
    import csv
    filepath = str(tmp_path / "test_log.csv")
    config = {"lr": 0.1, "weight_decay": 5e-4, "dropout": 0.1, "augmentation": "crop+flip"}
    log_experiment(filepath, "test_exp", config, 0.95, 0.90, "test note")
    with open(filepath) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert len(rows) == 1
    assert rows[0]["experiment"] == "test_exp"


# ---------------------------------------------------------------------------
# Split / selection discipline
#
# These use a stub dataset rather than the real CIFAR-10: a test suite should
# not need a 170 MB download to check that a split is disjoint.
# ---------------------------------------------------------------------------

class _FakeCIFAR(torch.utils.data.Dataset):
    """Stands in for torchvision's CIFAR10 with the same shapes and sizes."""

    #: Overridden by the tiny fixture so the training-loop test stays fast.
    TRAIN_N, TEST_N = 50000, 10000

    def __init__(self, root=None, train=True, download=False, transform=None):
        self.n = self.TRAIN_N if train else self.TEST_N
        self.transform = transform

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        g = torch.Generator().manual_seed(i)
        return torch.randn(3, 32, 32, generator=g), int(i % 10)


class _TinyCIFAR(_FakeCIFAR):
    """Module level so DataLoader workers can pickle it.

    A real-sized stub would train 4 epochs over 45,000 images on CPU; the
    property under test is bookkeeping, not scale.
    """
    TRAIN_N, TEST_N = 320, 64


@pytest.fixture
def stub_cifar(monkeypatch):
    import torchvision
    from src import train as train_mod
    monkeypatch.setattr(torchvision.datasets, "CIFAR10", _FakeCIFAR)
    monkeypatch.setattr(train_mod.torchvision.datasets, "CIFAR10", _FakeCIFAR)
    return train_mod


def test_validation_split_is_disjoint_from_training(stub_cifar):
    """Selection must not run on the test set.

    Regression test for the project's central defect: there was no validation
    split at all. The loop evaluated on TEST every 10 epochs, kept the best
    test accuracy it ever saw, early-stopped on it, and reported that same
    number -- a maximum over ~20 looks at the data it was being judged on.
    """
    trainloader, valloader, testloader = stub_cifar.get_data_loaders()

    assert len(trainloader.dataset) == 45000
    assert len(valloader.dataset) == 5000
    assert len(testloader.dataset) == 10000

    train_idx = set(trainloader.dataset.indices)
    val_idx = set(valloader.dataset.indices)
    assert not (train_idx & val_idx), "an image is in both train and validation"
    assert len(train_idx | val_idx) == 50000, "the split lost or duplicated images"


def test_validation_is_not_augmented(stub_cifar):
    """Augmentation is a training-time regulariser.

    Measuring on randomly cropped and flipped images makes validation noisy and
    pessimistic, and the noise lands directly on checkpoint selection.
    """
    trainloader, valloader, _ = stub_cifar.get_data_loaders(use_augmentation=True)
    assert valloader.dataset.dataset.transform is stub_cifar.transform_test
    assert trainloader.dataset.dataset.transform is stub_cifar.transform_augmented


def test_split_is_deterministic(stub_cifar):
    """A validation set that reshuffles between runs cannot compare experiments."""
    _, val_a, _ = stub_cifar.get_data_loaders()
    _, val_b, _ = stub_cifar.get_data_loaders()
    assert set(val_a.dataset.indices) == set(val_b.dataset.indices)


def test_training_returns_the_best_checkpoint_not_the_last(stub_cifar):
    """The reported accuracy must belong to the weights that get saved.

    `best_acc = max(best_acc, acc)` recorded a NUMBER while the function
    returned whatever state the model ended in -- so the headline figure and
    the saved checkpoint could come from different epochs. The best state_dict
    is now deep-copied on improvement and restored before returning.
    """
    import copy
    from src import train as train_mod

    import torchvision
    monkeypatch_targets = (torchvision.datasets, train_mod.torchvision.datasets)
    originals = [t.CIFAR10 for t in monkeypatch_targets]
    for t in monkeypatch_targets:
        t.CIFAR10 = _TinyCIFAR

    seen = {}
    real_eval = train_mod.evaluate

    # Validation accuracy that peaks in the middle, then degrades.
    scores = iter([10.0, 90.0, 20.0, 15.0])

    def fake_evaluate(model, loader, device):
        try:
            score = next(scores)
        except StopIteration:
            return 5.0
        if score == 90.0:
            seen["peak_weights"] = copy.deepcopy(model.state_dict())
        return score

    train_mod.evaluate = fake_evaluate
    try:
        model, _, _, _, best_val = train_mod.train(
            num_epochs=4, use_skip=False, use_augmentation=False,
            patience=99, val_size=64, num_workers=0)
    finally:
        train_mod.evaluate = real_eval
        for t, orig in zip(monkeypatch_targets, originals):
            t.CIFAR10 = orig

    assert best_val == 90.0, "did not report the peak validation score"
    final = model.state_dict()
    for k, v in seen["peak_weights"].items():
        assert torch.equal(final[k], v), (
            "returned model is not the checkpoint validation selected")
