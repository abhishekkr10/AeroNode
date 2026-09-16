"""Unit tests for the Python Control Plane simulator and ZeroMQ publisher."""

import json
import time
import unittest

import zmq

from control_plane.simulator import (
    DEFAULT_TOPIC,
    RuntimeState,
    SCHEMA_VERSION,
    StateGenerator,
    ZeroMQStatePublisher,
)


class ControlPlaneTests(unittest.TestCase):

    def test_valid_state_roundtrip(self):
        state = RuntimeState(
            schema_version=SCHEMA_VERSION,
            sequence=10,
            timestamp_unix_ms=1789500000000,
            cpu_load=45.5,
            temperature_c=62.0,
            latency_budget_ms=25.0,
            resource_pressure=0.45,
        )
        json_str = state.to_json()
        reconstructed = RuntimeState.from_json(json_str)
        self.assertEqual(state, reconstructed)

    def test_validation_rejects_out_of_bounds(self):
        # Invalid schema version
        with self.assertRaises(ValueError):
            RuntimeState(99, 1, 1000, 50.0, 50.0, 20.0, 0.5).validate()

        # Negative sequence
        with self.assertRaises(ValueError):
            RuntimeState(SCHEMA_VERSION, -1, 1000, 50.0, 50.0, 20.0, 0.5).validate()

        # Negative cpu_load
        with self.assertRaises(ValueError):
            RuntimeState(SCHEMA_VERSION, 1, 1000, -5.0, 50.0, 20.0, 0.5).validate()

        # CPU load > 100
        with self.assertRaises(ValueError):
            RuntimeState(SCHEMA_VERSION, 1, 1000, 105.0, 50.0, 20.0, 0.5).validate()

        # Non-positive latency budget
        with self.assertRaises(ValueError):
            RuntimeState(SCHEMA_VERSION, 1, 1000, 50.0, 50.0, 0.0, 0.5).validate()

        # Resource pressure > 1.0
        with self.assertRaises(ValueError):
            RuntimeState(SCHEMA_VERSION, 1, 1000, 50.0, 50.0, 20.0, 1.2).validate()

    def test_scenarios_generate_valid_states(self):
        scenarios = ["normal", "thermal_throttle", "strict_deadline", "dynamic"]
        for scenario in scenarios:
            gen = StateGenerator(scenario=scenario)
            for _ in range(20):
                state = gen.generate()
                # Should not raise
                state.validate()
                self.assertGreaterEqual(state.cpu_load, 0.0)
                self.assertLessEqual(state.cpu_load, 100.0)
                self.assertGreater(state.latency_budget_ms, 0.0)
                self.assertGreaterEqual(state.resource_pressure, 0.0)
                self.assertLessEqual(state.resource_pressure, 1.0)

    def test_zeromq_pub_sub_loopback(self):
        test_endpoint = "tcp://127.0.0.1:5599"

        with ZeroMQStatePublisher(endpoint=test_endpoint) as publisher:
            sub_context = zmq.Context()
            sub_socket = sub_context.socket(zmq.SUB)
            sub_socket.setsockopt(zmq.RCVTIMEO, 2000)
            sub_socket.connect(test_endpoint)
            sub_socket.setsockopt_string(zmq.SUBSCRIBE, DEFAULT_TOPIC)

            # Allow subscription handshake to propagate from SUB to PUB
            time.sleep(0.3)

            state = RuntimeState(
                schema_version=SCHEMA_VERSION,
                sequence=42,
                timestamp_unix_ms=int(time.time() * 1000),
                cpu_load=75.0,
                temperature_c=78.5,
                latency_budget_ms=15.0,
                resource_pressure=0.72,
            )
            publisher.publish(state)

            frames = sub_socket.recv_multipart()
            self.assertEqual(len(frames), 2)
            topic = frames[0].decode("utf-8")
            payload = frames[1].decode("utf-8")

            self.assertEqual(topic, DEFAULT_TOPIC)
            received_state = RuntimeState.from_json(payload)
            self.assertEqual(received_state.sequence, 42)
            self.assertEqual(received_state.cpu_load, 75.0)

            sub_socket.close()
            sub_context.term()



if __name__ == "__main__":
    unittest.main()
