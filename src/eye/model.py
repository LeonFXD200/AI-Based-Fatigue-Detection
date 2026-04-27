"""
Small CNN for eye open/closed classification.

Input:  grayscale 64x64 eye crop (1 channel)
Output: 2-class logits  [open, closed]

This architecture is intentionally small (~600K parameters) so that:
- Training is fast on commodity hardware
- Inference is fast on CPU for the live demo
- The model size is comparable to the MLPs used for the other modalities,
  keeping the four-way comparison fair on parameter count.
"""
from __future__ import annotations

import torch
from torch import nn


class SmallCNN(nn.Module):
    """3-block convolutional network for binary eye-state classification."""

    def __init__(self, num_classes: int = 2, dropout: float = 0.3) -> None:
        super().__init__()

        # Feature extractor: 3 conv blocks, each halving spatial dims
        # 64x64 -> 32x32 -> 16x16 -> 8x8
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        # Classifier: flatten -> dropout -> 2-layer MLP
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x

    def count_parameters(self) -> int:
        """Return the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick sanity check
    model = SmallCNN()
    dummy = torch.randn(4, 1, 64, 64)
    out = model(dummy)
    print(f"Model: SmallCNN")
    print(f"Parameters: {model.count_parameters():,}")
    print(f"Input shape:  {tuple(dummy.shape)}")
    print(f"Output shape: {tuple(out.shape)}")
