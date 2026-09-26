"""Step 3 — model factory (any timm CNN, ImageNet-pretrained)."""
import timm
import torch.nn as nn


def build_model(arch: str = "resnet50", pretrained: bool = True,
                n_classes: int = 2, dropout: float = 0.2) -> nn.Module:
    # timm replaces the 1000-class ImageNet head with a new n_classes head
    return timm.create_model(arch, pretrained=pretrained, num_classes=n_classes, drop_rate=dropout)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
