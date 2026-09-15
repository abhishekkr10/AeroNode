"""Deterministic Phase-1 model used to verify Python/ONNX/C++ parity."""

from __future__ import annotations

import torch
from torch import nn


REFERENCE_INPUT = torch.tensor([[-1.0, 0.5, 2.0, -0.25]], dtype=torch.float32)


class SimpleClassifier(nn.Module):
    """A tiny fixed-weight classifier, intentionally unrelated to model quality."""

    def __init__(self) -> None:
        super().__init__()
        self.hidden = nn.Linear(4, 5)
        self.activation = nn.ReLU()
        self.classifier = nn.Linear(5, 3)
        self._initialize_fixed_weights()

    def _initialize_fixed_weights(self) -> None:
        with torch.no_grad():
            self.hidden.weight.copy_(
                torch.tensor(
                    [
                        [0.10, 0.20, -0.10, 0.30],
                        [-0.20, 0.40, 0.30, -0.10],
                        [0.50, -0.30, 0.20, 0.10],
                        [0.25, 0.15, -0.35, 0.20],
                        [-0.40, 0.10, 0.25, 0.35],
                    ],
                    dtype=torch.float32,
                )
            )
            self.hidden.bias.copy_(
                torch.tensor([0.10, -0.20, 0.05, 0.30, -0.10], dtype=torch.float32)
            )
            self.classifier.weight.copy_(
                torch.tensor(
                    [
                        [0.30, -0.20, 0.40, 0.10, -0.10],
                        [-0.10, 0.50, -0.30, 0.20, 0.40],
                        [0.20, 0.10, 0.15, -0.40, 0.30],
                    ],
                    dtype=torch.float32,
                )
            )
            self.classifier.bias.copy_(
                torch.tensor([0.05, -0.15, 0.20], dtype=torch.float32)
            )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.activation(self.hidden(inputs)))


def create_model() -> SimpleClassifier:
    model = SimpleClassifier()
    model.eval()
    return model
