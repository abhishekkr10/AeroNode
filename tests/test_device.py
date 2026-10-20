"""Host sensing math, platform parsing and live ZeroMQ/native integration."""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import unittest

from control_plane.device import DeviceSampler, linux_counters, publish
from tools.evidence import ROOT, image_evidence


class DeviceStateTests(unittest.TestCase):
    def test_cpu_delta_and_pressure_proxy(self):
        readings = iter([(1000, 300, 0.4), (1200, 350, 0.8), (1300, 375, 0.2)])
        sampler = DeviceSampler(lambda: next(readings))
        result = sampler.sample(1, 30, 60)
        self.assertEqual(result["state"]["cpu_load"], 75)
        self.assertEqual(result["state"]["resource_pressure"], 0.8)
        self.assertEqual(result["provenance"]["temperature_c"], "configured_fallback_not_sensor")
        self.assertEqual(sampler.sample(2, 30, 60)["state"]["resource_pressure"], 0.75)

    def test_invalid_counter_deltas_fail_closed(self):
        for latest in ((100, 10, 0.5), (110, 35, 0.5), (110, 0, 0.5),
                       (110, 11, float("nan")), (110, 11, 1.5)):
            readings = iter([(100, 10, 0.5), latest])
            with self.subTest(latest=latest), self.assertRaises(ValueError):
                DeviceSampler(lambda: next(readings)).sample(1, 30, 60)

    def test_linux_guest_time_not_double_counted(self):
        with tempfile.TemporaryDirectory() as temporary:
            proc = Path(temporary)
            (proc / "stat").write_text("cpu 100 20 30 400 10 2 3 4 50 7\n")
            (proc / "meminfo").write_text("MemTotal: 1000 kB\nMemAvailable: 250 kB\n")
            self.assertEqual(linux_counters(proc), (569, 410, 0.75))

    def test_invalid_publisher_configuration(self):
        for frequency, count, deadline, temperature in ((0, 1, 30, 60), (float("nan"), 1, 30, 60),
                (5, 0, 30, 60), (5, 1, -1, 60), (5, 1, 30, float("inf"))):
            with self.subTest(frequency=frequency, count=count), self.assertRaises(ValueError):
                publish("tcp://127.0.0.1:5555", frequency, count, deadline, temperature, Path("unused"))


@unittest.skipUnless(os.environ.get("AERONODE_EVIDENCE_INTEGRATION") == "1", "Opt-in native integration")
class DeviceNativeTests(unittest.TestCase):
    def test_measured_publisher_reaches_native_snapshot(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            audit = directory / "state.jsonl"
            with socket.socket() as available:
                available.bind(("127.0.0.1", 0))
                port = available.getsockname()[1]
            endpoint = f"tcp://127.0.0.1:{port}"
            # File output prevents child stdout from blocking during native model loading.
            with (directory / "publisher.log").open("w") as log:
                process = subprocess.Popen([sys.executable, "-m", "control_plane.device", "--audit", str(audit),
                    "--endpoint", endpoint, "--frequency", "20", "--count", "600"], cwd=ROOT, stdout=log, stderr=log)
                try:
                    limit = time.monotonic() + 10
                    while not audit.exists() or audit.stat().st_size == 0:
                        if process.poll() is not None or time.monotonic() > limit:
                            self.fail("Publisher failed to produce audit records")
                        time.sleep(0.05)
                    image = next((ROOT / "data/imagenette2-160/val/n01440764").glob("*.JPEG"))
                    executable = Path(os.environ.get("AERONODE_RUNTIME_EXECUTABLE",
                        str(ROOT / "cmake-build-phase5e/runtime/aeronode_runtime.exe")))
                    report = image_evidence(image, ROOT / "model/artifacts/phase5", executable, None,
                        directory / "report", 30, endpoint)
                    state = report["adaptive"]["runtime_state"]
                    self.assertFalse(state["is_stale"])
                    records = [json.loads(line) for line in audit.read_text().splitlines()]
                    matched = next(record for record in records if record["state"]["sequence"] == state["sequence"])
                    self.assertEqual(matched["state"]["timestamp_unix_ms"], state["timestamp_unix_ms"])
                    self.assertAlmostEqual(matched["state"]["cpu_load"], state["cpu_load"], places=4)
                    self.assertEqual(report["state_source"], "external_ipc_check_publisher_audit")
                finally:
                    if process.poll() is None:
                        process.terminate()
                    process.wait(timeout=10)


if __name__ == "__main__":
    unittest.main()
