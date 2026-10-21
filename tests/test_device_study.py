"""Study integrity, bounded-worker cleanup and live native benchmark gates."""
import copy
import json
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

from benchmarks.device_study import ROOT, burn, environment, report, study, validate_observations


def fixture(label="device-a"):
    mode = {"requests": 2, "unique_samples": 2, "accuracy_pct": 50, "p95_latency_ms": 2, "mean_latency_ms": 1.5, "exit_distribution": {"4": 2},
            "deadline_misses": 0, "reference_max_abs_error": 0}
    return {"schema_version": 1, "device_label": label, "status": "completed",
        "protocol": {"rounds": 1, "samples": 2, "repeats": 1}, "contract": {"input_sha256": "test"},
        "cases": [{"round": 1, "condition": condition, "observations": {"cpu_load_mean_pct": 25},
            "modes": {name: copy.deepcopy(mode) for name in ("monolithic_full", "adaptive_current", "static_current")}}
            for condition in ("idle", "contention")]}


class StudyContractTests(unittest.TestCase):
    def test_compatible_devices_compare_without_pooling(self):
        text = report([fixture(), fixture("device-b")])
        self.assertIn("device-b", text)
        self.assertIn("Round results remain separate", text)

    def test_incompatible_inputs_and_protocol_rejected(self):
        for key in ("contract", "protocol"):
            modified = fixture("device-b")
            modified[key]["different"] = True
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "Incompatible"):
                report([fixture(), modified])

    def test_incomplete_failed_and_duplicate_cases_rejected(self):
        for change in ("failed", "missing", "duplicate"):
            data = fixture()
            if change == "failed": data["status"] = "failed"
            if change == "missing": data["cases"].pop()
            if change == "duplicate": data["cases"].append(data["cases"][0])
            with self.subTest(change=change), self.assertRaises(ValueError): report([data])
        with self.assertRaisesRegex(ValueError, "distinct"):
            report([fixture(), fixture()])

    def test_invalid_summary_cannot_be_compared(self):
        for key, value in (("p95_latency_ms", float("nan")), ("accuracy_pct", 101),
                           ("reference_max_abs_error", 0.01), ("requests", 1), ("deadline_misses", 3)):
            data = fixture()
            data["cases"][0]["modes"]["adaptive_current"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): report([data])

    def test_burn_is_bounded(self):
        for seconds, duty in ((0, 0.5), (901, 0.5), (1, 1), (1, float("nan"))):
            with self.assertRaises(ValueError): burn(seconds, duty)

    def test_children_terminated_on_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "state.jsonl").write_text("{}\n" * 5)
            worker, publisher = Mock(), Mock()
            worker.poll.return_value = publisher.poll.return_value = None
            with patch("benchmarks.device_study.subprocess.Popen", side_effect=[worker, publisher]), patch("benchmarks.device_study.terminate") as stop:
                with self.assertRaisesRegex(RuntimeError, "intentional"):
                    with environment(directory, 1, 0.5, 30, 3):
                        raise RuntimeError("intentional")
                self.assertEqual(stop.call_count, 2)

    def test_audit_mismatch_and_stale_observations_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            state = {"sequence": 1, "timestamp_unix_ms": 1000, "cpu_load": 25,
                     "resource_pressure": 0.5, "temperature_c": 60, "is_stale": False}
            audit = dict(state, latency_budget_ms=3)
            (directory / "state.jsonl").write_text(json.dumps({"state": audit}) + "\n")
            row = {"runtime_state": state, "state_source": "external_ipc", "repetition": 0,
                   "sample_index": 0, "latency_budget_ms": 3}
            def save():
                (directory / "records.jsonl").write_text(json.dumps(row) + "\n")
            save()
            self.assertEqual(validate_observations(directory)["paired_observations"], 1)
            for key, value in (("cpu_load", 75), ("is_stale", True), ("sequence", 2)):
                row["runtime_state"] = dict(state, **{key: value})
                save()
                with self.subTest(key=key), self.assertRaises(ValueError): validate_observations(directory)


@unittest.skipUnless(os.environ.get("AERONODE_EVIDENCE_INTEGRATION") == "1", "Opt-in native study")
class StudyNativeTests(unittest.TestCase):
    def test_idle_contention_study_with_actual_state_and_parity(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "study"
            executable = Path(os.environ.get("AERONODE_BENCHMARK_EXECUTABLE",
                str(ROOT / "cmake-build-phase5e/runtime/aeronode_benchmark.exe")))
            data = study(output, "local-test", executable, samples=10, repeats=1, warmup=1,
                         rounds=1, workers=1, timeout=30)
            self.assertEqual(data["status"], "completed")
            self.assertEqual(len(data["cases"]), 2)
            for case in data["cases"]:
                self.assertEqual(case["observations"]["paired_observations"], 10)
                self.assertEqual(case["modes"]["staged_exit_1"]["total_stage_runs"], [10, 0, 0, 0])
                self.assertLessEqual(case["modes"]["adaptive_current"]["reference_max_abs_error"], 1e-5)
            with self.assertRaises(FileExistsError):
                study(output, "local-test", executable)

    def test_native_missing_publisher_fails_without_records(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "records.jsonl"
            with socket.socket() as available:
                available.bind(("127.0.0.1", 0))
                port = available.getsockname()[1]
            exe = os.environ.get("AERONODE_BENCHMARK_EXECUTABLE", str(ROOT / "cmake-build-phase5e/runtime/aeronode_benchmark.exe"))
            result = subprocess.run([exe, "--input-bin", "unused", "--labels-bin", "unused",
                "--stage-dir", "unused", "--monolithic-model", "unused", "--records", str(output),
                "--samples", "1", "--ipc-endpoint", f"tcp://127.0.0.1:{port}"],
                capture_output=True, text=True, timeout=10)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("No fresh benchmark IPC state", result.stderr)
            self.assertFalse(output.exists())
