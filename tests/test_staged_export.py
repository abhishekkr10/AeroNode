"""Automated verification test for Phase 5A staged ONNX exports and manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

import onnx
from onnx import shape_inference
import onnxruntime as ort

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PHASE5_DIR = PROJECT_ROOT / "model" / "artifacts" / "phase5"
MANIFEST_PATH = PHASE5_DIR / "staged_export_manifest.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


class StagedExportTests(unittest.TestCase):
    """Verify structural integrity and contract validity of Phase 5A artifacts."""

    def test_manifest_and_files_exist(self) -> None:
        self.assertTrue(MANIFEST_PATH.is_file(), f"Manifest missing: {MANIFEST_PATH}")
        with MANIFEST_PATH.open("r", encoding="utf-8") as f:
            manifest = json.load(f)

        self.assertEqual(manifest.get("aeronode_phase"), "5A")
        self.assertEqual(len(manifest.get("stages", [])), 4)

        for stage in manifest["stages"]:
            stage_path = PROJECT_ROOT / stage["relative_path"]
            self.assertTrue(stage_path.is_file(), f"Stage file missing: {stage_path}")
            self.assertEqual(
                sha256_file(stage_path),
                stage["sha256"],
                f"SHA256 mismatch for {stage['file_name']}",
            )

    def test_onnx_checker_and_strict_shapes(self) -> None:
        expected_contracts = [
            (
                "stage_1.onnx",
                {"input": ["batch_size", 3, 128, 128]},
                {"feat_1": ["batch_size", 32, 32, 32], "exit_1_logits": ["batch_size", 10]},
            ),
            (
                "stage_2.onnx",
                {"feat_1": ["batch_size", 32, 32, 32]},
                {"feat_2": ["batch_size", 64, 16, 16], "exit_2_logits": ["batch_size", 10]},
            ),
            (
                "stage_3.onnx",
                {"feat_2": ["batch_size", 64, 16, 16]},
                {"feat_3": ["batch_size", 128, 8, 8], "exit_3_logits": ["batch_size", 10]},
            ),
            (
                "stage_4.onnx",
                {"feat_3": ["batch_size", 128, 8, 8]},
                {"exit_4_logits": ["batch_size", 10]},
            ),
        ]

        for file_name, expected_inputs, expected_outputs in expected_contracts:
            stage_path = PHASE5_DIR / file_name
            self.assertTrue(stage_path.is_file())

            model = onnx.load(stage_path)
            # Full check
            onnx.checker.check_model(model, full_check=True)

            # Strict shape inference
            inferred = shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)

            # Check inputs
            inputs = {}
            for inp in inferred.graph.input:
                dims = []
                for d in inp.type.tensor_type.shape.dim:
                    dims.append(d.dim_value if d.HasField("dim_value") else d.dim_param)
                inputs[inp.name] = dims
                self.assertEqual(inp.type.tensor_type.elem_type, onnx.TensorProto.FLOAT)
            self.assertEqual(inputs, expected_inputs)

            # Check outputs
            outputs = {}
            for out in inferred.graph.output:
                dims = []
                for d in out.type.tensor_type.shape.dim:
                    dims.append(d.dim_value if d.HasField("dim_value") else d.dim_param)
                outputs[out.name] = dims
                self.assertEqual(out.type.tensor_type.elem_type, onnx.TensorProto.FLOAT)
            self.assertEqual(outputs, expected_outputs)

    def test_onnx_runtime_loadability(self) -> None:
        for index in range(1, 5):
            stage_path = PHASE5_DIR / f"stage_{index}.onnx"
            session = ort.InferenceSession(str(stage_path), providers=["CPUExecutionProvider"])
            self.assertGreater(len(session.get_inputs()), 0)
            self.assertGreater(len(session.get_outputs()), 0)

    def test_analytical_flops_breakdown(self) -> None:
        with MANIFEST_PATH.open("r", encoding="utf-8") as f:
            manifest = json.load(f)

        summary = manifest["compute_summary"]
        self.assertEqual(summary["total_analytical_flops"], 315368872)
        self.assertEqual(summary["total_analytical_macs"], 157684416)

        stage_flops_sum = sum(s["theoretical_flops"] for s in manifest["stages"])
        self.assertEqual(stage_flops_sum, summary["total_analytical_flops"])


if __name__ == "__main__":
    unittest.main()
