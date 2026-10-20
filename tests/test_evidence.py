"""Evidence workflow failure gates and real native image execution."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from tools.evidence import (ROOT, SYNSETS, digest, doctor, image_evidence, pack,
                            preprocess, render_report, run_native, verify_stages)


class EvidenceContractTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.stages = self.root / "stages"
        self.stages.mkdir()
        self.manifest = {"source_checkpoint": {"sha256": "a" * 64,
            "class_to_idx": {s: i for i, s in enumerate(SYNSETS)}}, "stages": []}
        for i in range(1, 5):
            path = self.stages / f"stage_{i}.onnx"
            path.write_bytes(bytes([i]))
            self.manifest["stages"].append({"stage_name": f"stage_{i}",
                "file_name": path.name, "sha256": digest(path)})
        self.save()

    def save(self):
        (self.stages / "staged_export_manifest.json").write_text(json.dumps(self.manifest), encoding="utf-8")

    def test_pack_roundtrip_and_no_overwrite(self):
        destination = self.root / "transfer"
        pack(self.stages, destination)
        self.assertEqual(verify_stages(destination), self.manifest)
        with self.assertRaises(FileExistsError):
            pack(self.stages, destination)

    def test_modified_model_rejected_before_pack(self):
        (self.stages / "stage_2.onnx").write_bytes(b"modified")
        with self.assertRaisesRegex(ValueError, "Integrity mismatch"):
            pack(self.stages, self.root / "transfer")
        self.assertFalse((self.root / "transfer").exists())

    def test_incorrect_label_mapping_rejected(self):
        self.manifest["source_checkpoint"]["class_to_idx"][SYNSETS[0]] = 9
        self.save()
        with self.assertRaisesRegex(ValueError, "class mapping"):
            verify_stages(self.stages)

    def test_manifest_path_traversal_rejected(self):
        self.manifest["stages"][0]["file_name"] = "../outside.onnx"
        self.save()
        with self.assertRaisesRegex(ValueError, "file name"):
            verify_stages(self.stages)

    def test_missing_model_rejected(self):
        (self.stages / "stage_4.onnx").unlink()
        with self.assertRaises(FileNotFoundError):
            verify_stages(self.stages)

    def test_wrong_policy_checkpoint_rejected(self):
        executable = self.root / "runtime.exe"
        executable.write_bytes(b"placeholder")
        policy = self.root / "policy.json"
        policy.write_text(json.dumps({"source_checkpoint_sha256": "b" * 64}))
        with self.assertRaisesRegex(ValueError, "checkpoint"):
            doctor(self.stages, executable, policy)

    def test_invalid_deadline_and_existing_output_rejected(self):
        for deadline in (0, -1, float("nan"), float("inf")):
            with self.assertRaisesRegex(ValueError, "Deadline"):
                image_evidence(self.root / "missing", self.stages, self.root / "runtime", None,
                               self.root / "result", deadline)
        with self.assertRaises(FileExistsError):
            image_evidence(self.root / "missing", self.stages, self.root / "runtime", None, self.root, 30)

    def test_native_failure_not_retried(self):
        with patch("tools.evidence.subprocess.run") as run:
            run.return_value.returncode = 1
            run.return_value.stderr = "dependency missing"
            with self.assertRaisesRegex(RuntimeError, "dependency missing"):
                run_native(self.root / "runtime", self.stages, self.root / "input", None, 30)
            self.assertEqual(run.call_count, 1)

    def test_invalid_execution_counters_rejected(self):
        with patch("tools.evidence.subprocess.run") as run:
            run.return_value.returncode = 0
            run.return_value.stdout = json.dumps({"selected_exit": 1, "stage_run_counts": [1, 1, 1, 1]})
            with self.assertRaisesRegex(ValueError, "execution prefix"):
                run_native(self.root / "runtime", self.stages, self.root / "input", None, 30)

    def test_stale_external_snapshot_rejected(self):
        with patch("tools.evidence.subprocess.run") as run:
            run.return_value.returncode = 0
            run.return_value.stdout = json.dumps({"runtime_state": {"is_stale": True}})
            with self.assertRaisesRegex(ValueError, "No fresh IPC state"):
                run_native(self.root / "runtime", self.stages, self.root / "input", None, 30,
                           ipc_endpoint="tcp://127.0.0.1:5555")


@unittest.skipUnless(os.environ.get("AERONODE_EVIDENCE_INTEGRATION") == "1",
                     "Set AERONODE_EVIDENCE_INTEGRATION=1 with local artifacts for native checks")
class EvidenceNativeTests(unittest.TestCase):
    def setUp(self):
        self.image = next((ROOT / "data/imagenette2-160/val/n01440764").glob("*.JPEG"))
        self.executable = Path(os.environ.get("AERONODE_RUNTIME_EXECUTABLE",
            str(ROOT / "cmake-build-phase5e/runtime/aeronode_runtime.exe")))

    def test_preprocessing_matches_benchmark_dataset(self):
        from model.training.imagenette_data import create_datasets
        _, dataset = create_datasets(ROOT / "data/imagenette2-160")
        index = next(i for i, (path, _) in enumerate(dataset.samples) if Path(path) == self.image)
        expected, _ = dataset[index]
        np.testing.assert_array_equal(preprocess(self.image)[0], expected.numpy())

    def test_image_native_parity_and_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "result"
            report = image_evidence(self.image, ROOT / "model/artifacts/phase5", self.executable,
                ROOT / "runtime/config/calibrated_imagenette_v1.json", destination, 30)
            self.assertLessEqual(report["selected_head_max_abs_error"], 1e-5)
            self.assertEqual(report["full_depth"]["stage_run_counts"], [1, 1, 1, 1])
            saved = json.loads((destination / "evidence.json").read_text())
            self.assertEqual(saved, report)
            self.assertIn("not a compute or energy percentage", (destination / "report.html").read_text(encoding="utf-8"))
            changed = copy.deepcopy(report)
            changed["created_utc"] = "<script>alert(1)</script>"
            self.assertNotIn("<script>", render_report(changed))

    def test_tight_budget_proves_downstream_skip(self):
        with tempfile.TemporaryDirectory() as temporary:
            report = image_evidence(self.image, ROOT / "model/artifacts/phase5", self.executable,
                None, Path(temporary) / "result", 0.001)
            self.assertEqual(report["adaptive"]["stage_run_counts"], [1, 0, 0, 0])
            self.assertEqual(report["adaptive"]["stages"][-1]["sample_evaluations"][0]["reason"],
                             "LATENCY_BUDGET_EXHAUSTED")


if __name__ == "__main__":
    unittest.main()
