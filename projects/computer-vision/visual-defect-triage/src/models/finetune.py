"""Optional full fine-tune, for when the linear probe plateaus."""


def build_optimiser(backbone, head, base_lr: float = 1e-3, ratio: int = 100):
    """Discriminative learning rates, a hundred to one by default.

    The pretrained weights are the asset you are buying. Training them at the
    head's rate overwrites general visual features with whatever this small
    dataset contains, and the result is worse than leaving them frozen.
    """
    import torch

    return torch.optim.AdamW(
        [
            {"params": backbone.parameters(), "lr": base_lr / ratio},
            {"params": head.parameters(), "lr": base_lr},
        ],
        weight_decay=1e-4,
    )
