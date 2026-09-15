"""Compact ResNet-style multi-exit classifier for Imagenette."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F


DEFAULT_EXIT_WEIGHTS = (0.15, 0.20, 0.25, 0.40)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut: nn.Module
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(inputs)
        outputs = F.relu(self.bn1(self.conv1(inputs)), inplace=True)
        outputs = self.bn2(self.conv2(outputs))
        return F.relu(outputs + residual, inplace=True)


class ExitHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(in_channels, num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(torch.flatten(self.pool(features), 1))


class MultiExitResNet(nn.Module):
    """Four-stage ResNet-18-style backbone with a classifier after each stage."""

    def __init__(
        self,
        num_classes: int = 10,
        channels: Sequence[int] = (32, 64, 128, 256),
        blocks_per_stage: Sequence[int] = (2, 2, 2, 2),
    ) -> None:
        super().__init__()
        if len(channels) != 4 or len(blocks_per_stage) != 4:
            raise ValueError("MultiExitResNet requires exactly four stages")
        if any(value <= 0 for value in (*channels, *blocks_per_stage, num_classes)):
            raise ValueError("Channel, block, and class counts must be positive")

        self.num_classes = num_classes
        self.channels = tuple(channels)
        self.blocks_per_stage = tuple(blocks_per_stage)
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        in_channels = channels[0]
        stages: list[nn.Module] = []
        for stage_index, (out_channels, block_count) in enumerate(
            zip(channels, blocks_per_stage, strict=True)
        ):
            stride = 1 if stage_index == 0 else 2
            blocks: list[nn.Module] = [BasicBlock(in_channels, out_channels, stride)]
            blocks.extend(BasicBlock(out_channels, out_channels) for _ in range(block_count - 1))
            stages.append(nn.Sequential(*blocks))
            in_channels = out_channels
        self.stages = nn.ModuleList(stages)
        self.exit_heads = nn.ModuleList(
            ExitHead(channel_count, num_classes) for channel_count in channels
        )
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.zeros_(module.bias)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        features = self.stem(inputs)
        logits: list[torch.Tensor] = []
        for stage, head in zip(self.stages, self.exit_heads, strict=True):
            features = stage(features)
            logits.append(head(features))
        return tuple(logits)


def weighted_multi_exit_loss(
    logits: Sequence[torch.Tensor],
    targets: torch.Tensor,
    weights: Sequence[float] = DEFAULT_EXIT_WEIGHTS,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    if len(logits) != len(weights) or not logits:
        raise ValueError("There must be one positive loss weight per exit")
    if any(weight <= 0 for weight in weights):
        raise ValueError("Exit loss weights must be positive")
    weight_sum = float(sum(weights))
    normalized_weights = tuple(weight / weight_sum for weight in weights)
    exit_losses = tuple(F.cross_entropy(exit_logits, targets) for exit_logits in logits)
    total = sum(
        weight * exit_loss
        for weight, exit_loss in zip(normalized_weights, exit_losses, strict=True)
    )
    return total, exit_losses
