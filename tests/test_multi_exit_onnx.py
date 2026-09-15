"""Automated four-output ONNX structure and parity tests."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import torch

from model.export.export_multi_exit import (
    OUTPUT_NAMES,
    compare_outputs,
    export_four_output_model,
    validate_graph,
)
from model.training.multi_exit_resnet import MultiExitResNet


class MultiExitOnnxTests(unittest.TestCase):
    def test_four_output_export_and_parity(self) -> None:
        torch.manual_seed(19)
        model = MultiExitResNet(
            num_classes=10,
            channels=(8, 16, 32, 64),
            blocks_per_stage=(1, 1, 1, 1),
        ).eval()
        inputs = torch.randn(2, 3, 128, 128)
        with tempfile.TemporaryDirectory() as temporary_directory:
            model_path = Path(temporary_directory) / "multi_exit_test.onnx"
            export_four_output_model(model, inputs[:1], model_path)
            graph = validate_graph(model_path)
            comparisons = compare_outputs(model, model_path, inputs)

        self.assertEqual(list(graph["outputs"]), list(OUTPUT_NAMES))
        self.assertEqual(len(comparisons), 4)
        self.assertTrue(all(item["shape"] == [2, 10] for item in comparisons))
        self.assertTrue(all(item["parity"] for item in comparisons))


if __name__ == "__main__":
    unittest.main()
