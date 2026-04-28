"""
Small MLP for steering-based fatigue classification.

Input:  engineered driving telemetry features per window
Output: 2-class logits  [normal, drowsy]

Same MLP architecture as the face modality (per the dissertation
methodology section requiring consistent ML techniques across modalities).
"""
from __future__ import annotations

import torch
from torch import nn


class SteeringMLP(nn.Module):
    """Small MLP for binary drowsy classification from driving telemetry."""

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
    model = SteeringMLP(n_features=10)
    dummy = torch.randn(4, 10)
    out = model(dummy)
    print(f"Model: SteeringMLP")
    print(f"Parameters: {model.count_parameters():,}")
    print(f"Input shape:  {tuple(dummy.shape)}")
    print(f"Output shape: {tuple(out.shape)}")
