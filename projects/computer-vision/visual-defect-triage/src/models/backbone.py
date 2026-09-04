"""The ViT backbone. One forward pass per image, cached by version."""
EMBEDDING_VERSION = "vitb16-augreg2-v3"


class Backbone:
    """Wraps timm, falling back to torchvision if timm is not installed.

    num_classes=0 is the whole trick: it removes the head and returns the pooled
    768-d feature, which is what all three consumers read.
    """

    def __init__(self, device: str | None = None):
        import torch

        from src.config import settings

        self.device = device or settings.device
        try:
            import timm

            self.net = timm.create_model(settings.backbone, pretrained=True, num_classes=0)
        except ImportError:
            from torchvision.models import ViT_B_16_Weights, vit_b_16

            net = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
            net.heads = torch.nn.Identity()
            self.net = net
        self.net.eval().to(self.device)
        self._torch = torch

    def __call__(self, pixel_values):
        """(B, 3, 224, 224) -> (B, 768)"""
        with self._torch.inference_mode():
            return self.net(pixel_values.to(self.device))

    @property
    def version(self) -> str:
        return EMBEDDING_VERSION
