"""ConvNeXtV2 feature extractor for perceptual loss, using timm."""

import torch
import torch.nn as nn
import timm

from utils.logging_util import log_for_0


class ConvNextV2Features(nn.Module):
    """ConvNeXtV2-Base feature extractor using timm pretrained weights."""

    def __init__(self, model_name: str = "convnextv2_base.fcmae_ft_in22k_in1k"):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=True, num_classes=0)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract global-average-pooled features.

        Args:
            x: (B, C, H, W) in any value range (will be resized internally
               by timm's data config if needed).
        Returns:
            features: (B, feature_dim)
        """
        return self.model(x)


_cached_model: ConvNextV2Features | None = None


def load_convnext_model(device: torch.device | str = "cuda") -> ConvNextV2Features:
    """Load (and cache) the ConvNeXtV2 feature extractor."""
    global _cached_model
    if _cached_model is None:
        log_for_0("Loading ConvNeXtV2 feature extractor via timm...")
        _cached_model = ConvNextV2Features()
        param_count = sum(p.numel() for p in _cached_model.parameters())
        log_for_0(f"ConvNeXtV2 loaded with {param_count:,} parameters.")
    model = _cached_model.to(device)
    return model
