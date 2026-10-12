"""Functional Phase-5D CLI checks, including the existing ZeroMQ control plane."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tests.test_phase4_integrated import LivePublisherThread

EXECUTABLE = Path(os.environ.get("AERONODE_RUNTIME_EXECUTABLE", str(ROOT / "build/runtime/aeronode_runtime.exe")))
FIXTURES = ROOT / "tmp/phase5d"


class StagedExecutionCliTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not EXECUTABLE.is_file():
            raise FileNotFoundError(EXECUTABLE)
        if not (FIXTURES / "input_5.bin").is_file():
            subprocess.run([sys.executable, str(ROOT / "tests/prepare_phase5d_fixtures.py")], check=True)

    def run_request(self, batch=1, extra=(), with_input=True, expected_success=True):
        command = [str(EXECUTABLE), "--staged-dir", str(ROOT / "model/artifacts/phase5"),
                   "--batch-size", str(batch)]
        if with_input:
            command += ["--input-bin", str(FIXTURES / f"input_{batch}.bin")]
        result = subprocess.run(command + list(extra), cwd=ROOT, capture_output=True, text=True, timeout=30)
        if expected_success:
            self.assertEqual(result.returncode, 0, result.stderr)
            return json.loads(result.stdout)
        self.assertNotEqual(result.returncode, 0, "Invalid request unexpectedly succeeded")
        return result.stderr

    def check_forced_exit(self, exit):
        thresholds = [-1.0] * 4
        if exit < 4:
            thresholds[exit - 1] = 3.0
        for batch in (1, 3, 5):
            with self.subTest(exit=exit, batch=batch):
                result = self.run_request(batch, ["--latency-budget", "1000000", "--resource-pressure", "0",
                    "--entropy-thresholds", ",".join(map(str, thresholds))])
                self.assertEqual(result["selected_exit"], exit)
                self.assertEqual(result["shape"], [batch, 10])
                self.assertEqual(result["stage_run_counts"], [int(stage < exit) for stage in range(4)])
                self.assertEqual(len(result["stages"]), exit)
                expected = np.fromfile(FIXTURES / f"reference_{batch}_{exit}.bin", dtype=np.float32)
                actual = np.array(result["logits"], dtype=np.float32)
                self.assertLessEqual(float(np.max(np.abs(expected - actual))), 1e-5)
                np.testing.assert_array_equal(actual.reshape(batch, 10).argmax(1), expected.reshape(batch, 10).argmax(1))
                self.assertNotIn("average_latency_ms", result)

    def test_forced_exit_1(self):
        self.check_forced_exit(1)

    def test_forced_exit_2(self):
        self.check_forced_exit(2)

    def test_forced_exit_3(self):
        self.check_forced_exit(3)

    def test_forced_exit_4(self):
        self.check_forced_exit(4)

    def test_requires_preprocessed_input(self):
        self.assertIn("requires --input-bin", self.run_request(with_input=False, expected_success=False))

    def test_rejects_wrong_input_file_length(self):
        path = FIXTURES / "short_input.bin"
        path.write_bytes(b"1234")
        self.assertIn("size mismatch", self.run_request(extra=["--input-bin", str(path)], expected_success=False))

    def test_rejects_invalid_runtime_state(self):
        self.assertIn("Invalid RuntimeState", self.run_request(extra=["--latency-budget", "0"], expected_success=False))

    def test_rejects_staged_benchmark_iterations(self):
        self.assertIn("one request", self.run_request(extra=["--iterations", "2"], expected_success=False))

    def test_model_modes_are_exclusive(self):
        self.assertIn("Usage", self.run_request(extra=["--model", str(ROOT / "model/artifacts/simple_model.onnx")],
                                                expected_success=False))

    def test_conservative_snapshot_without_ipc(self):
        result = self.run_request(extra=["--entropy-thresholds", "3,3,3,3"])
        self.assertTrue(result["runtime_state"]["is_stale"])
        self.assertEqual(result["selected_exit"], 1)
        self.assertEqual(result["stage_run_counts"], [1, 0, 0, 0])

    def test_live_ipc_snapshot_with_staged_inference(self):
        endpoint = "tcp://127.0.0.1:5566"
        publisher = LivePublisherThread(endpoint=endpoint, frequency_hz=30.0)
        publisher.start()
        time.sleep(0.3)
        try:
            result = self.run_request(extra=["--ipc-endpoint", endpoint, "--entropy-thresholds", "3,3,3,3"])
            self.assertGreater(result["runtime_state"]["sequence"], 0)
            self.assertFalse(result["runtime_state"]["is_stale"])
            self.assertGreater(result["runtime_state"]["cpu_load"], 70)
            self.assertEqual(result["stage_run_counts"], [1, 0, 0, 0])
        finally:
            publisher.stop()


if __name__ == "__main__":
    unittest.main()
