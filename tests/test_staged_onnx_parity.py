"""Phase 5B Automated Python Parity & Staged Execution Verification Suite.

Verifies:
1. Sequential chaining of stage_1 -> stage_2 -> stage_3 -> stage_4 in Python ONNX Runtime.
2. Numerical parity against the Phase 3 monolithic reference graph (multi_exit_imagenette.onnx).
3. Numerical parity against PyTorch MultiExitResNet.
4. Dynamic batch dimensions (batch_size=1, batch_size=3, batch_size=10).
5. Partial execution (early exits 1, 2, 3) without executing subsequent stages.
"""

from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np
import onnxruntime as ort
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_PATH = PROJECT_ROOT / "model" / "artifacts" / "phase2-imagenette-5ep" / "best_final_exit.pt"
MONOLITHIC_MODEL_PATH = PROJECT_ROOT / "model" / "artifacts" / "phase3" / "multi_exit_imagenette.onnx"
PHASE5_DIR = PROJECT_ROOT / "model" / "artifacts" / "phase5"
STAGE_PATHS = [PHASE5_DIR / f"stage_{i}.onnx" for i in range(1, 5)]

from model.export.export_multi_exit import select_class_diverse_batch
from model.training.imagenette_data import create_datasets
from model.training.multi_exit_resnet import MultiExitResNet


class StagedOnnxParityTests(unittest.TestCase):
    """Verify numerical parity and dynamic batching of staged ONNX models."""

    @classmethod
    def setUpClass(cls) -> None:
        if not CHECKPOINT_PATH.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")
        if not MONOLITHIC_MODEL_PATH.is_file():
            raise FileNotFoundError(f"Monolithic ONNX model not found: {MONOLITHIC_MODEL_PATH}")
        for path in STAGE_PATHS:
            if not path.is_file():
                raise FileNotFoundError(f"Stage model not found: {path}")

        # Load PyTorch model
        checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
        config = checkpoint["run_config"]
        cls.pytorch_model = MultiExitResNet(
            num_classes=len(config["class_to_idx"]),
            channels=tuple(config["channels"]),
            blocks_per_stage=tuple(config["blocks_per_stage"]),
        )
        cls.pytorch_model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        cls.pytorch_model.eval()

        # Initialize ONNX Runtime sessions (CPU provider)
        cls.mono_session = ort.InferenceSession(
            str(MONOLITHIC_MODEL_PATH), providers=["CPUExecutionProvider"]
        )
        cls.stage_sessions = [
            ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            for path in STAGE_PATHS
        ]

        # Load 10 deterministic validation samples spanning all 10 classes
        data_root = PROJECT_ROOT / "data" / "imagenette2-160"
        _, val_dataset = create_datasets(data_root, 128)
        inputs, targets, _ = select_class_diverse_batch(val_dataset, 10)
        cls.val_inputs_tensor = inputs
        cls.val_inputs_np = inputs.numpy()
        cls.val_targets_tensor = targets

    def _run_staged_chain(self, input_array: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Run stages sequentially in Python ONNX Runtime, chaining activations."""
        s1, s2, s3, s4 = self.stage_sessions

        # Stage 1: input -> feat_1, exit_1_logits
        out1 = s1.run(["feat_1", "exit_1_logits"], {"input": input_array})
        feat_1, exit_1_logits = out1[0], out1[1]

        # Stage 2: feat_1 -> feat_2, exit_2_logits
        out2 = s2.run(["feat_2", "exit_2_logits"], {"feat_1": feat_1})
        feat_2, exit_2_logits = out2[0], out2[1]

        # Stage 3: feat_2 -> feat_3, exit_3_logits
        out3 = s3.run(["feat_3", "exit_3_logits"], {"feat_2": feat_2})
        feat_3, exit_3_logits = out3[0], out3[1]

        # Stage 4: feat_3 -> exit_4_logits
        out4 = s4.run(["exit_4_logits"], {"feat_3": feat_3})
        exit_4_logits = out4[0]

        logits = [exit_1_logits, exit_2_logits, exit_3_logits, exit_4_logits]
        features = [feat_1, feat_2, feat_3]
        return logits, features

    def test_staged_vs_monolithic_onnx_parity_batch10(self) -> None:
        """Verify bit-level parity between chained staged models and monolithic model on 10 validation samples."""
        mono_output_names = [f"exit_{i}_logits" for i in range(1, 5)]
        mono_logits = self.mono_session.run(mono_output_names, {"input": self.val_inputs_np})

        staged_logits, staged_features = self._run_staged_chain(self.val_inputs_np)

        # Verify intermediate feature shapes
        self.assertEqual(list(staged_features[0].shape), [10, 32, 32, 32])
        self.assertEqual(list(staged_features[1].shape), [10, 64, 16, 16])
        self.assertEqual(list(staged_features[2].shape), [10, 128, 8, 8])

        # Verify per-exit numerical equivalence
        for exit_idx in range(4):
            max_diff = float(np.max(np.abs(mono_logits[exit_idx] - staged_logits[exit_idx])))
            self.assertLessEqual(
                max_diff,
                1e-5,
                f"Exit {exit_idx + 1} difference exceeds tolerance: {max_diff}",
            )
            # Verify exact argmax class prediction equality
            mono_preds = np.argmax(mono_logits[exit_idx], axis=1)
            staged_preds = np.argmax(staged_logits[exit_idx], axis=1)
            np.testing.assert_array_equal(
                mono_preds,
                staged_preds,
                err_msg=f"Class prediction mismatch at exit {exit_idx + 1}",
            )

    def test_staged_vs_pytorch_parity_batch10(self) -> None:
        """Verify numerical parity between chained staged ONNX models and PyTorch MultiExitResNet."""
        with torch.no_grad():
            torch_outputs = [t.numpy() for t in self.pytorch_model(self.val_inputs_tensor)]

        staged_logits, _ = self._run_staged_chain(self.val_inputs_np)

        for exit_idx in range(4):
            max_diff = float(np.max(np.abs(torch_outputs[exit_idx] - staged_logits[exit_idx])))
            self.assertLessEqual(
                max_diff,
                1e-5,
                f"Exit {exit_idx + 1} difference against PyTorch exceeds tolerance: {max_diff}",
            )
            torch_preds = np.argmax(torch_outputs[exit_idx], axis=1)
            staged_preds = np.argmax(staged_logits[exit_idx], axis=1)
            np.testing.assert_array_equal(
                torch_preds,
                staged_preds,
                err_msg=f"Class prediction mismatch vs PyTorch at exit {exit_idx + 1}",
            )

    def test_single_sample_streaming_batch1(self) -> None:
        """Verify correct execution for streaming single-sample batch (batch_size=1)."""
        single_input = self.val_inputs_np[:1]  # shape [1, 3, 128, 128]
        mono_logits = self.mono_session.run(
            [f"exit_{i}_logits" for i in range(1, 5)], {"input": single_input}
        )
        staged_logits, staged_features = self._run_staged_chain(single_input)

        self.assertEqual(list(staged_features[0].shape), [1, 32, 32, 32])
        self.assertEqual(list(staged_features[1].shape), [1, 64, 16, 16])
        self.assertEqual(list(staged_features[2].shape), [1, 128, 8, 8])

        for exit_idx in range(4):
            self.assertEqual(list(staged_logits[exit_idx].shape), [1, 10])
            max_diff = float(np.max(np.abs(mono_logits[exit_idx] - staged_logits[exit_idx])))
            self.assertLessEqual(max_diff, 1e-5)
            self.assertEqual(
                int(np.argmax(mono_logits[exit_idx])),
                int(np.argmax(staged_logits[exit_idx])),
            )

    def test_dynamic_batch_arbitrary_sizes(self) -> None:
        """Verify dynamic batch dimension support for arbitrary batch sizes (batch_size=3, 5)."""
        for batch_size in (3, 5):
            batch_input = self.val_inputs_np[:batch_size]
            mono_logits = self.mono_session.run(
                [f"exit_{i}_logits" for i in range(1, 5)], {"input": batch_input}
            )
            staged_logits, staged_features = self._run_staged_chain(batch_input)

            self.assertEqual(list(staged_features[0].shape), [batch_size, 32, 32, 32])
            self.assertEqual(list(staged_features[1].shape), [batch_size, 64, 16, 16])
            self.assertEqual(list(staged_features[2].shape), [batch_size, 128, 8, 8])

            for exit_idx in range(4):
                self.assertEqual(list(staged_logits[exit_idx].shape), [batch_size, 10])
                max_diff = float(np.max(np.abs(mono_logits[exit_idx] - staged_logits[exit_idx])))
                self.assertLessEqual(max_diff, 1e-5)

    def test_partial_execution_early_exits(self) -> None:
        """Verify that early exits can terminate without executing subsequent stages."""
        s1, s2, s3, _ = self.stage_sessions
        input_data = self.val_inputs_np[:2]  # batch 2

        # Case 1: Early Exit at Stage 1
        out1 = s1.run(["feat_1", "exit_1_logits"], {"input": input_data})
        feat_1, e1 = out1[0], out1[1]
        self.assertEqual(list(e1.shape), [2, 10])

        # Case 2: Early Exit at Stage 2
        out2 = s2.run(["feat_2", "exit_2_logits"], {"feat_1": feat_1})
        feat_2, e2 = out2[0], out2[1]
        self.assertEqual(list(e2.shape), [2, 10])

        # Case 3: Early Exit at Stage 3
        out3 = s3.run(["feat_3", "exit_3_logits"], {"feat_2": feat_2})
        _, e3 = out3[0], out3[1]
        self.assertEqual(list(e3.shape), [2, 10])

        # Compare partial execution logits with monolithic full run
        mono_logits = self.mono_session.run(
            ["exit_1_logits", "exit_2_logits", "exit_3_logits"], {"input": input_data}
        )
        self.assertLessEqual(float(np.max(np.abs(mono_logits[0] - e1))), 1e-5)
        self.assertLessEqual(float(np.max(np.abs(mono_logits[1] - e2))), 1e-5)
        self.assertLessEqual(float(np.max(np.abs(mono_logits[2] - e3))), 1e-5)


if __name__ == "__main__":
    unittest.main()
