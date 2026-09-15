"""Structural and supervision tests for the Phase-2 model."""

from __future__ import annotations

import unittest

import torch

from model.training.multi_exit_resnet import MultiExitResNet, weighted_multi_exit_loss


class MultiExitResNetTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.model = MultiExitResNet(
            num_classes=10, channels=(8, 16, 32, 64), blocks_per_stage=(1, 1, 1, 1)
        )
        self.inputs = torch.randn(3, 3, 64, 64)
        self.targets = torch.tensor([0, 4, 9])

    def test_all_four_heads_have_expected_shape(self) -> None:
        outputs = self.model(self.inputs)
        self.assertEqual(len(outputs), 4)
        for output in outputs:
            self.assertEqual(tuple(output.shape), (3, 10))

    def test_each_head_receives_direct_supervision(self) -> None:
        outputs = self.model(self.inputs)
        total_loss, exit_losses = weighted_multi_exit_loss(outputs, self.targets)
        self.assertEqual(len(exit_losses), 4)
        total_loss.backward()
        for head in self.model.exit_heads:
            gradient = head.classifier.weight.grad
            self.assertIsNotNone(gradient)
            self.assertGreater(float(torch.linalg.vector_norm(gradient)), 0.0)

    def test_loss_rejects_mismatched_weights(self) -> None:
        outputs = self.model(self.inputs)
        with self.assertRaises(ValueError):
            weighted_multi_exit_loss(outputs, self.targets, weights=(0.5, 0.5))


if __name__ == "__main__":
    unittest.main()
