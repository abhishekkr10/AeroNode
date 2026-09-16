"""AeroNode Python Control Plane: RuntimeState Simulator and ZeroMQ Publisher."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
import time
from typing import Optional

import zmq

SCHEMA_VERSION = 1
DEFAULT_TOPIC = "runtime_state"
DEFAULT_ENDPOINT = "tcp://127.0.0.1:5555"
DEFAULT_FREQUENCY_HZ = 5.0


@dataclass(frozen=True)
class RuntimeState:
    schema_version: int
    sequence: int
    timestamp_unix_ms: int
    cpu_load: float
    temperature_c: float
    latency_budget_ms: float
    resource_pressure: float

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"Unsupported schema_version: {self.schema_version}")
        if self.sequence < 0:
            raise ValueError(f"Sequence must be non-negative: {self.sequence}")
        if not (0.0 <= self.cpu_load <= 100.0):
            raise ValueError(f"cpu_load out of range [0, 100]: {self.cpu_load}")
        if self.latency_budget_ms <= 0.0:
            raise ValueError(f"latency_budget_ms must be positive: {self.latency_budget_ms}")
        if not (0.0 <= self.resource_pressure <= 1.0):
            raise ValueError(f"resource_pressure out of range [0, 1]: {self.resource_pressure}")

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))

    @classmethod
    def from_json(cls, payload: str | bytes) -> RuntimeState:
        data = json.loads(payload)
        state = cls(
            schema_version=int(data["schema_version"]),
            sequence=int(data["sequence"]),
            timestamp_unix_ms=int(data["timestamp_unix_ms"]),
            cpu_load=float(data["cpu_load"]),
            temperature_c=float(data["temperature_c"]),
            latency_budget_ms=float(data["latency_budget_ms"]),
            resource_pressure=float(data["resource_pressure"]),
        )
        state.validate()
        return state


class StateGenerator:
    """Generates synthetic RuntimeState sequences based on configurable scenarios."""

    def __init__(self, scenario: str = "dynamic"):
        self.scenario = scenario
        self._sequence = 0

    def generate(self, timestamp_unix_ms: Optional[int] = None) -> RuntimeState:
        if timestamp_unix_ms is None:
            timestamp_unix_ms = int(time.time() * 1000)

        seq = self._sequence
        self._sequence += 1
        t_sec = (timestamp_unix_ms / 1000.0) % 3600.0

        if self.scenario == "normal":
            cpu_load = 25.0 + 5.0 * math.sin(t_sec * 0.2)
            temperature_c = 48.0 + 2.0 * math.sin(t_sec * 0.1)
            latency_budget_ms = 45.0
            resource_pressure = 0.20 + 0.05 * math.sin(t_sec * 0.2)

        elif self.scenario == "thermal_throttle":
            cpu_load = 88.0 + 7.0 * math.sin(t_sec * 0.5)
            temperature_c = 86.0 + 4.0 * math.sin(t_sec * 0.3)
            latency_budget_ms = 18.0
            resource_pressure = 0.85 + 0.10 * math.sin(t_sec * 0.5)

        elif self.scenario == "strict_deadline":
            cpu_load = 92.0
            temperature_c = 82.0
            latency_budget_ms = 8.0
            resource_pressure = 0.95

        elif self.scenario == "dynamic":
            # Oscillates between benign and stressed conditions
            phase = math.sin(t_sec * 0.2)
            cpu_load = 50.0 + 40.0 * phase
            temperature_c = 60.0 + 25.0 * phase
            latency_budget_ms = 25.0 - 15.0 * phase  # Tight budget when load is high
            resource_pressure = 0.50 + 0.45 * phase

        else:
            raise ValueError(f"Unknown scenario: {self.scenario}")

        # Clamp all values to legal boundaries
        cpu_load = max(0.0, min(100.0, cpu_load))
        resource_pressure = max(0.0, min(1.0, resource_pressure))
        latency_budget_ms = max(1.0, latency_budget_ms)

        state = RuntimeState(
            schema_version=SCHEMA_VERSION,
            sequence=seq,
            timestamp_unix_ms=timestamp_unix_ms,
            cpu_load=round(cpu_load, 2),
            temperature_c=round(temperature_c, 2),
            latency_budget_ms=round(latency_budget_ms, 2),
            resource_pressure=round(resource_pressure, 3),
        )
        state.validate()
        return state


class ZeroMQStatePublisher:
    """Publishes RuntimeState frames over a ZeroMQ PUB socket."""

    def __init__(self, endpoint: str = DEFAULT_ENDPOINT, topic: str = DEFAULT_TOPIC):
        self.endpoint = endpoint
        self.topic = topic
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        # Avoid hanging on close if subscribers are slow
        self.socket.setsockopt(zmq.LINGER, 100)
        self.socket.bind(self.endpoint)

    def publish(self, state: RuntimeState) -> None:
        state.validate()
        topic_frame = self.topic.encode("utf-8")
        payload_frame = state.to_json().encode("utf-8")
        self.socket.send_multipart([topic_frame, payload_frame])

    def close(self) -> None:
        self.socket.close()
        self.context.term()

    def __enter__(self) -> ZeroMQStatePublisher:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


def run_simulator(
    endpoint: str = DEFAULT_ENDPOINT,
    frequency_hz: float = DEFAULT_FREQUENCY_HZ,
    scenario: str = "dynamic",
    max_count: Optional[int] = None,
    duration_sec: Optional[float] = None,
) -> None:
    generator = StateGenerator(scenario=scenario)
    interval_sec = 1.0 / frequency_hz
    start_time = time.time()
    count = 0

    print(f"[AeroNode Control Plane] Binding ZeroMQ PUB to {endpoint}")
    print(f"[AeroNode Control Plane] Scenario: {scenario}, Frequency: {frequency_hz} Hz")

    with ZeroMQStatePublisher(endpoint=endpoint) as publisher:
        # Give subscribers 150ms to establish connection if already waiting
        time.sleep(0.15)
        try:
            while True:
                state = generator.generate()
                publisher.publish(state)
                count += 1
                if count % 10 == 0 or max_count is not None:
                    print(
                        f"[PUB #{state.sequence:04d}] CPU: {state.cpu_load:5.1f}% | "
                        f"Temp: {state.temperature_c:5.1f} C | Budget: {state.latency_budget_ms:5.1f} ms | "
                        f"Pressure: {state.resource_pressure:.3f}"
                    )

                if max_count is not None and count >= max_count:
                    break
                if duration_sec is not None and (time.time() - start_time) >= duration_sec:
                    break

                time.sleep(interval_sec)
        except KeyboardInterrupt:
            print("\n[AeroNode Control Plane] Stopped by user.")


def main() -> None:
    parser = argparse.ArgumentParser(description="AeroNode Control Plane Telemetry Simulator")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT, help="ZeroMQ PUB endpoint")
    parser.add_argument("--frequency", type=float, default=DEFAULT_FREQUENCY_HZ, help="Publish rate in Hz")
    parser.add_argument(
        "--scenario",
        choices=["normal", "thermal_throttle", "strict_deadline", "dynamic"],
        default="dynamic",
        help="Simulation scenario",
    )
    parser.add_argument("--count", type=int, default=None, help="Maximum number of states to emit")
    parser.add_argument("--duration", type=float, default=None, help="Run duration in seconds")
    args = parser.parse_args()

    run_simulator(
        endpoint=args.endpoint,
        frequency_hz=args.frequency,
        scenario=args.scenario,
        max_count=args.count,
        duration_sec=args.duration,
    )


if __name__ == "__main__":
    main()
