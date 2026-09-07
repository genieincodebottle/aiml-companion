"""One transform definition, shared by training, evaluation, and the API.

The API imports eval_transform from here rather than reimplementing it. That is
the single highest-value line in the project: when serving resizes differently
from validation, accuracy drops a point or two and every offline number stays
green.
"""
from src.config import settings

# ImageNet defaults, used only if timm cannot report the real ones. Hardcoding
# these when the checkpoint wants something else is a classic silent loss.
_FALLBACK_MEAN = (0.485, 0.456, 0.406)
_FALLBACK_STD = (0.229, 0.224, 0.225)


def normalisation():
    """Ask timm for the exact constants the pretrained weights were trained with."""
    try:
        import timm

        cfg = timm.get_pretrained_cfg(settings.backbone)
        return tuple(cfg.mean), tuple(cfg.std)
    except Exception:
        return _FALLBACK_MEAN, _FALLBACK_STD


def eval_transform():
    """Used by validation, test, index building AND the API. One definition."""
    from torchvision import transforms as T

    mean, std = normalisation()
    return T.Compose([
        T.Resize(settings.image_size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(settings.image_size),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])


def train_transform():
    """Augmentation must add variation the camera can actually produce.

    No vertical flip and no rotation: on a line the part is always the same way
    up, so those teach a variation that never occurs and cost accuracy on the
    ones that do.
    """
    from torchvision import transforms as T

    mean, std = normalisation()
    return T.Compose([
        T.RandomResizedCrop(settings.image_size, scale=(0.7, 1.0),
                            interpolation=T.InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.2, contrast=0.2),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
