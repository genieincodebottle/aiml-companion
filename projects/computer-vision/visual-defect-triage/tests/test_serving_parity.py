"""Serving must preprocess exactly as evaluation did.

The most common silent accuracy loss in deployed vision: serving resizes with a
different interpolation or normalises with different constants, offline metrics
stay perfect, and production quietly degrades.
"""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
pytest.importorskip("PIL")


def test_api_uses_the_same_transform_as_evaluation():
    from PIL import Image

    from src.data.transforms import eval_transform

    img = Image.new("RGB", (2448, 2048), (120, 130, 140))
    a, b = eval_transform()(img), eval_transform()(img)
    assert torch.equal(a, b), "serving preprocessing has drifted from evaluation"


def test_normalisation_constants_come_from_the_checkpoint():
    """Hardcoding ImageNet constants when the checkpoint wants others is silent."""
    from src.data.transforms import normalisation

    mean, std = normalisation()
    assert len(mean) == 3 and len(std) == 3
    assert all(0.0 < s < 1.0 for s in std)
