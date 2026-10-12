"""Comprehensive Phase 4 end-to-end integration test.

Verifies:
1. Python Control Plane publishing RuntimeState over ZeroMQ IPC.
2. C++ Aeronode Runtime receiving live telemetry in background thread.
3. Accurate multi-exit ONNX execution across all 4 exits.
4. Correct Shannon entropy and confidence calculation.
5. Dynamic early-exit decision rules (confidence, latency budget, resource pressure).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from control_plane.simulator import (
    RuntimeState,
    SCHEMA_VERSION,
    StateGenerator,
    ZeroMQStatePublisher,
)

EXECUTABLE = Path(os.environ.get("AERONODE_RUNTIME_EXECUTABLE", str(PROJECT_ROOT / "build" / "runtime" / "aeronode_runtime.exe")))
MODEL_PATH = PROJECT_ROOT / "model" / "artifacts" / "phase3" / "multi_exit_imagenette.onnx"
TEST_IPC_ENDPOINT = "tcp://127.0.0.1:5566"


class LivePublisherThread(threading.Thread):
    def __init__(self, endpoint: str, scenario: str = "thermal_throttle", frequency_hz: float = 20.0):
        super().__init__(daemon=True)
        self.endpoint = endpoint
        self.scenario = scenario
        self.frequency_hz = frequency_hz
        self.stop_event = threading.Event()
        self.generator = StateGenerator(scenario=scenario)
        self.publisher = None

    def run(self):
        with ZeroMQStatePublisher(endpoint=self.endpoint) as pub:
            self.publisher = pub
            interval = 1.0 / self.frequency_hz
            while not self.stop_event.is_set():
                state = self.generator.generate()
                pub.publish(state)
                time.sleep(interval)

    def stop(self):
        self.stop_event.set()
        self.join(timeout=2.0)


class Phase4IntegratedTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not EXECUTABLE.exists():
            raise FileNotFoundError(f"Executable not found: {EXECUTABLE}")
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    def run_runtime(self, extra_args: list[str]) -> dict:
        cmd = [
            str(EXECUTABLE.resolve()),
            "--model",
            str(MODEL_PATH.resolve()),
        ] + extra_args
        completed = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )
        return json.loads(completed.stdout)

    def test_live_ipc_state_reception_and_inference(self):
        """Verify C++ runtime connects to Python Control Plane over ZeroMQ and acquires live state."""
        publisher = LivePublisherThread(endpoint=TEST_IPC_ENDPOINT, scenario="thermal_throttle", frequency_hz=30.0)
        publisher.start()
        # Allow publisher to bind and emit initial messages
        time.sleep(0.3)

        try:
            result = self.run_runtime(
                [
                    "--ipc-endpoint",
                    TEST_IPC_ENDPOINT,
                    "--iterations",
                    "5",
                ]
            )

            # 1. Check runtime state snapshot
            self.assertIn("runtime_state", result)
            state = result["runtime_state"]
            self.assertEqual(state["schema_version"], SCHEMA_VERSION)
            self.assertGreater(state["sequence"], 0)
            self.assertFalse(state["is_stale"])
            self.assertGreater(state["cpu_load"], 70.0)  # Thermal throttle scenario has high CPU
            self.assertGreater(state["temperature_c"], 75.0)

            # 2. Check 4 exits
            self.assertIn("exits", result)
            self.assertEqual(len(result["exits"]), 4)
            for idx, exit_entry in enumerate(result["exits"], start=1):
                self.assertEqual(exit_entry["name"], f"exit_{idx}_logits")
                self.assertEqual(len(exit_entry["predictions"]), 1)
                pred = exit_entry["predictions"][0]
                self.assertIn("entropy", pred)
                self.assertIn("confidence", pred)
                self.assertGreaterEqual(pred["confidence"], 0.1)
                self.assertLessEqual(pred["confidence"], 1.0)
                self.assertGreaterEqual(pred["entropy"], 0.0)

            # 3. Check decision
            self.assertIn("decision", result)
            decision = result["decision"]
            self.assertIn(decision["selected_exit"], [1, 2, 3, 4])
            self.assertIn(
                decision["reason"],
                [
                    "CONFIDENCE_MET",
                    "LATENCY_BUDGET_EXHAUSTED",
                    "RESOURCE_PRESSURE_FORCED_EXIT",
                    "FINAL_EXIT_REACHED",
                ],
            )
            self.assertTrue(len(decision["evaluations"]) >= 1)

        finally:
            publisher.stop()

    def test_decision_latency_budget_exhausted(self):
        """Verify exit triggered when latency budget is too strict."""
        result = self.run_runtime(
            [
                "--iterations",
                "1",
                "--latency-budget",
                "4.0",
                "--entropy-thresholds",
                "0.1,0.1,0.1,2.5",
            ]
        )
        decision = result["decision"]
        self.assertEqual(decision["selected_exit"], 1)
        self.assertEqual(decision["reason"], "LATENCY_BUDGET_EXHAUSTED")
        self.assertEqual(decision["evaluations"][0]["action"], "EXIT")

    def test_decision_confidence_early_exit(self):
        """Verify early exit on exit 1 when entropy threshold is generous."""
        result = self.run_runtime(
            [
                "--iterations",
                "1",
                "--latency-budget",
                "50.0",
                "--entropy-thresholds",
                "1.8,2.0,2.0,2.5",
            ]
        )
        decision = result["decision"]
        self.assertEqual(decision["selected_exit"], 1)
        self.assertEqual(decision["reason"], "CONFIDENCE_MET")
        self.assertEqual(decision["evaluations"][0]["action"], "EXIT")

    def test_decision_final_exit_reached(self):
        """Verify fallback to final exit when intermediate heads are not confident."""
        result = self.run_runtime(
            [
                "--iterations",
                "1",
                "--latency-budget",
                "50.0",
                "--entropy-thresholds",
                "0.1,0.1,0.1,2.5",
            ]
        )
        decision = result["decision"]
        self.assertEqual(decision["selected_exit"], 4)
        self.assertEqual(decision["reason"], "FINAL_EXIT_REACHED")
        evals = decision["evaluations"]
        self.assertEqual(len(evals), 4)
        self.assertEqual(evals[0]["action"], "CONTINUE")
        self.assertEqual(evals[1]["action"], "CONTINUE")
        self.assertEqual(evals[2]["action"], "CONTINUE")
        self.assertEqual(evals[3]["action"], "EXIT")


if __name__ == "__main__":
    unittest.main()
