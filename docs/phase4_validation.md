# Phase 4 Validation Record

## Overview

Phase 4 integrates the validated Phase 3 four-output ONNX model (`model/artifacts/phase3/multi_exit_imagenette.onnx`) with a C++17 inference runtime, a Python Control Plane publishing real-time telemetry over ZeroMQ IPC, a numerically stable Shannon entropy calculator, and a multi-exit early-termination decision engine.

All Phase 0–3 work, tests, and models were preserved without regression.

---

## Verified Components

### 1. Multi-Exit C++17 ONNX Runtime Engine
- Dynamic batch execution for input `input` shaped `[batch, 3, 128, 128]`.
- Verified all four output tensors and shapes:
  - `exit_1_logits`: `[batch, 10]`
  - `exit_2_logits`: `[batch, 10]`
  - `exit_3_logits`: `[batch, 10]`
  - `exit_4_logits`: `[batch, 10]`
- Numerical parity verified on the 10-class validation batch against Python ONNX Runtime:
  - Maximum absolute error across all exits: `5.9604645e-08` (tolerance `atol=1e-5`, `rtol=1e-4`).
  - Predicted class agreement across all 4 exits: 100%.

### 2. Shannon Entropy & Confidence Module (`runtime/include/entropy.hpp`)
- Numerically stable softmax with max subtraction:
  $$p_i = \frac{\exp(z_i - \max z)}{\sum_j \exp(z_j - \max z)}$$
- Shannon entropy in natural units (nats) using log-sum-exp formulation to avoid underflow:
  $$H(p) = \ln\left(\sum_j \exp(z_j - \max z)\right) - \sum_i p_i (z_i - \max z)$$
- Unit tests (`tests/test_entropy.cpp`):
  - Uniform distribution ($z_i = c$): $H = \ln(10) \approx 2.302585$, confidence = 0.10.
  - One-hot peaked distribution: $H \approx 0.0$, confidence = 1.00.
  - Extreme dynamic range: logits up to $+10000.0$ without overflow.
  - Extreme negative values: logits down to $-5000.0$ without underflow/NaN.
  - Arbitrary distribution: exact mathematical match against manual derivation.

### 3. Python Control Plane Simulator & ZeroMQ Publisher (`control_plane/simulator.py`)
- Wire protocol: UTF-8 JSON matching schema version 1, multipart ZeroMQ with topic `runtime_state`.
- Fields: `schema_version`, `sequence`, `timestamp_unix_ms`, `cpu_load` (0..100), `temperature_c`, `latency_budget_ms` (> 0), `resource_pressure` (0..1).
- Pinned dependency: `pyzmq==27.2.0` in `requirements-phase4.txt`.
- Scenarios: `normal`, `thermal_throttle`, `strict_deadline`, `dynamic`.
- Tested in `tests/test_control_plane.py`: serialization roundtrip, boundary validation, and loopback PUB/SUB.

### 4. Thread-Safe C++ ZeroMQ State Receiver (`runtime/include/state_receiver.hpp`)
- Non-blocking background thread running dedicated SUB socket loop with atomic stop token.
- Mutex-protected `RuntimeState` snapshot storage (`GetLatestSnapshot()`).
- Error handling: validates JSON schema and bounds; drops out-of-order and malformed messages.
- Staleness detection: marks state stale and falls back to conservative default if elapsed time > TTL (2000 ms).
- Tested in `tests/test_state_receiver.cpp`: lifecycle, fallback, live message flow, drop counters, and clean shutdown.

### 5. Early-Exit Decision Engine (`runtime/include/exit_decision.hpp`)
- Sequential stage evaluation ($k \in \{1, 2, 3, 4\}$).
- Decision reasons:
  - `CONFIDENCE_MET`: intermediate entropy $\le$ calibrated threshold.
  - `LATENCY_BUDGET_EXHAUSTED`: projected inference latency exceeds `latency_budget_ms`.
  - `RESOURCE_PRESSURE_FORCED_EXIT`: elevated `resource_pressure` ($\ge 0.80$) with acceptable confidence.
  - `FINAL_EXIT_REACHED`: fallback to Exit 4.
- Tested in `tests/test_exit_decision.cpp` and integrated in `aeronode_runtime`.

---

## Automated Test Results

| Test Target | Suite | Cases | Result |
|---|---|---|---|
| Phase 2 Backbone & Supervised Loss | `tests/test_multi_exit_model.py` | 3 | Pass |
| Phase 3 ONNX Export & Graph Parity | `tests/test_multi_exit_onnx.py` | 1 | Pass |
| Phase 4 Control Plane & Schema | `tests/test_control_plane.py` | 4 | Pass |
| Phase 4 End-to-End Integration | `tests/test_phase4_integrated.py` | 4 | Pass |
| Phase 4 C++ Shannon Entropy | `tests/test_entropy.cpp` | 5 | Pass |
| Phase 4 C++ ZeroMQ State Receiver | `tests/test_state_receiver.cpp` | 3 | Pass |
| Phase 4 C++ Exit Decision Engine | `tests/test_exit_decision.cpp` | 4 | Pass |
| Phase 1 C++ Parity Verifier | `tests/verify_phase1.py` | 1 | Pass |
| Phase 4 C++ 4-Exit Engine Parity | `tests/verify_phase4_engine.py` | 4 exits | Pass |

**Total Automated Tests**: 25/25 passed (100% success rate).

---

## Architectural Clarification: Monolithic ONNX Graph & Compute Savings

The Phase 3/4 ONNX model (`multi_exit_imagenette.onnx`) is a monolithic four-output graph. In Phase 4:
- The C++ runtime validates the structural integrity, numerical fidelity, entropy calculation, and sequential EXIT/CONTINUE decision logic across all four heads.
- Executing standard `session.Run` on this monolithic graph computes the shared backbone and heads; therefore, **no compute or latency savings are claimed for early-exit decisions on this monolithic artifact**.
- True compute savings and stage-level latency reductions require decomposing the model into separately executable stage graphs, as specified in `architecture.md`. Phase 4 establishes the verified decision and telemetry foundation for that subsequent work.

