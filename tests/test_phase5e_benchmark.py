"""Validate measurements, paired record accounting, and native benchmark failures."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from benchmarks.run_phase5e import select_indices
from benchmarks.summarize_phase5e import MODES, summarize

ROOT = Path(__file__).resolve().parents[1]
EXE = Path(os.environ.get("AERONODE_BENCHMARK_EXECUTABLE", str(ROOT / "cmake-build-phase5e/runtime/aeronode_benchmark.exe")))


def matrix():
    rows = []
    for repetition in range(2):
        for sample in range(2):
            for mode in MODES:
                exit = int(mode[-1]) if mode.startswith("staged_exit_") else 4
                mono = mode == "monolithic_full"
                rows.append(dict(mode=mode, sample_index=sample, repetition=repetition, label=sample,
                    predicted_class=0, selected_exit=exit, stage_run_counts=[0]*4 if mono else [int(i<exit) for i in range(4)],
                    monolithic_session_runs=int(mono), latency_ms=1.0+sample, reference_max_abs_error=0,
                    entropy=1.0, confidence=0.7, reason="FULL_DEPTH_BASELINE" if mono else "FINAL_EXIT_REACHED",
                    latency_budget_ms=30, deadline_missed=False))
    return rows


class BenchmarkSummaryTests(unittest.TestCase):
    def summarize_rows(self, rows):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.jsonl"
            path.write_text("".join(json.dumps(row)+"\n" for row in rows), encoding="utf-8")
            return summarize(path, 2, 2)

    def test_known_accuracy_latency_and_counts(self):
        result = self.summarize_rows(matrix())
        self.assertEqual(result["record_count"], 24)
        mode = result["modes"]["staged_exit_2"]
        self.assertEqual(mode["accuracy_pct"], 50)
        self.assertEqual(mode["mean_latency_ms"], 1.5)
        self.assertEqual(mode["p95_latency_ms"], 2)
        self.assertEqual(mode["total_stage_runs"], [4,4,0,0])

    def test_missing_request_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            self.summarize_rows(matrix()[:-1])

    def test_duplicate_request_is_rejected(self):
        rows = matrix()
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.summarize_rows(rows+[rows[0]])

    def test_wrong_execution_path_is_rejected(self):
        rows = matrix(); rows[1]["stage_run_counts"] = [1,1,1,1]
        with self.assertRaisesRegex(ValueError, "counter mismatch"):
            self.summarize_rows(rows)

    def test_bad_parity_and_nonfinite_measurements_are_rejected(self):
        for field, value in (("latency_ms", float("nan")), ("latency_ms", 0), ("reference_max_abs_error", 1.1e-5)):
            with self.subTest(field=field, value=value):
                rows = matrix(); rows[0][field] = value
                with self.assertRaisesRegex(ValueError, "timing or numerical"):
                    self.summarize_rows(rows)

    def test_paired_label_mismatch_is_rejected(self):
        rows = matrix(); rows[1]["label"] = 3
        with self.assertRaisesRegex(ValueError, "Inconsistent label"):
            self.summarize_rows(rows)

    def test_sampling_is_reproducible_and_balanced(self):
        labels = [i for i in range(10) for _ in range(4)]
        selected = select_indices(labels, 20, 17)
        self.assertEqual(selected, select_indices(labels, 20, 17))
        self.assertEqual(len(set(selected)), 20)
        self.assertEqual([sum(labels[i]==c for i in selected) for c in range(10)], [2]*10)
        self.assertEqual(set(select_indices(labels, 0, 17)), set(range(40)))
        with self.assertRaises(ValueError): select_indices(labels, 41, 17)


class NativeBenchmarkTests(unittest.TestCase):
    def test_paired_execution_parity_and_bad_files(self):
        if not EXE.is_file():
            raise FileNotFoundError(EXE)
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            labels = directory / "labels.bin"
            labels.write_bytes((0).to_bytes(4, "little", signed=True))
            command = [str(EXE), "--input-bin", str(ROOT / "tmp/phase5d/input_1.bin"),
                "--labels-bin", str(labels), "--samples", "1", "--warmup", "1", "--repeats", "1",
                "--stage-dir", str(ROOT / "model/artifacts/phase5"), "--monolithic-model",
                str(ROOT / "model/artifacts/phase3/multi_exit_imagenette.onnx"), "--records", str(directory / "records.jsonl")]
            completed = subprocess.run(command, capture_output=True, text=True, timeout=30)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            metadata = json.loads(completed.stdout)
            self.assertTrue(metadata["completed"])
            self.assertGreater(metadata["process_peak_working_set_bytes"], 0)
            result = summarize(directory / "records.jsonl", 1, 1)
            self.assertEqual(result["record_count"], 6)
            self.assertLessEqual(max(entry["reference_max_abs_error"] for entry in result["modes"].values()), 1e-5)
            repeat = subprocess.run(command, capture_output=True, text=True, timeout=30)
            self.assertNotEqual(repeat.returncode, 0)
            self.assertIn("overwrite", repeat.stderr)
            labels.write_bytes(b"broken")
            bad = subprocess.run(command, capture_output=True, text=True, timeout=30)
            self.assertNotEqual(bad.returncode, 0)
            self.assertIn("size mismatch", bad.stderr)


if __name__ == "__main__":
    unittest.main()
