"""
Small MLP for face-based fatigue classification.

Input:  engineered facial features per image (head pose, mouth ratio, etc.)
Output: 2-class logits  [alert, drowsy]

This architecture is intentionally small to be comparable in capacity
to the small CNN used for the eye modality, supporting fair four-way
comparison across modalities as described in the dissertation methodology.
"""
from __future__ import annotations

import torch
from torch import nn


class FaceMLP(nn.Module):
    """Small MLP for binary drowsiness classification from face features."""

    def __init__(self, n_features: int, num_classes: int = 2, dropout: float = 0.25) -> None:
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(n_features, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick sanity check
    model = FaceMLP(n_features=8)
    dummy = torch.randn(4, 8)
    out = model(dummy)
    print(f"Model: FaceMLP")
    print(f"Parameters: {model.count_parameters():,}")
    print(f"Input shape:  {tuple(dummy.shape)}")
    print(f"Output shape: {tuple(out.shape)}")
