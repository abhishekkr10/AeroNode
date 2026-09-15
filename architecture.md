# AeroNode Architecture

## Project Purpose

AeroNode is an experimental edge-AI inference runtime that changes neural-network inference depth in response to current resource constraints and model confidence. Its result must be a reproducible measurement of the accuracy, latency, and compute trade-off, not merely an early-exit model demonstration.

## Problem Statement and Core Engineering Question

Fixed-depth inference ignores changes in CPU load, thermal pressure, and latency requirements. AeroNode asks whether an adaptive early-exit policy can reduce edge-class latency and computational work while retaining acceptable accuracy under changing runtime constraints.

## Implementation State

This document separates committed design from implemented behavior. The Phase-1 single-model vertical slice is implemented and verified across PyTorch, ONNX, Python ONNX Runtime, and C++ ONNX Runtime. Multi-exit execution, IPC, policy adaptation, full benchmarks, allocator experiments, and telemetry are planned and are not yet implemented. `current_status.md` is the source of truth for completed work.

## High-Level Architecture

```text
Python control plane -- ZeroMQ PUB/SUB --> Adaptive C++ runtime
                                                |
                                                v
                                      ONNX Runtime staged models
                                                |
                                                v
                               prediction + decision + measurements
                                                |
                                                v
                                  FastAPI telemetry --> PostgreSQL
```

The Phase-1 vertical slice deliberately reduces this to PyTorch model definition/export -> one ONNX model -> C++17 ONNX Runtime inference -> output comparison and latency measurement.

## ASE Abstraction

- **Environment:** Python simulator that publishes changing CPU load, temperature, latency budget, and resource pressure.
- **Agent:** deterministic adaptive policy in C++ that combines the latest environment snapshot with entropy/confidence.
- **System:** C++17 runtime that executes ONNX stages, makes exit decisions, and measures outcomes.

## Python Control Plane

Python is outside the core inference path. It will provide model training/export, a resource-state simulator, benchmark orchestration, and later the FastAPI telemetry service. The simulator will publish at a configurable low frequency (initial target: 5 Hz), independent of neural-network layers or inference requests.

## RuntimeState Schema

The wire payload will be versioned UTF-8 JSON:

```json
{
  "schema_version": 1,
  "sequence": 42,
  "timestamp_unix_ms": 1789500000000,
  "cpu_load": 92.0,
  "temperature_c": 81.5,
  "latency_budget_ms": 15.0,
  "resource_pressure": 0.87
}
```

Ranges are `cpu_load` 0..100, `temperature_c` in degrees Celsius, positive `latency_budget_ms`, and `resource_pressure` 0..1. Invalid messages are rejected without replacing the last valid state. A state older than a configured TTL is marked stale and the policy falls back to a documented conservative state.

## ZeroMQ IPC Architecture

- **Pattern:** Python PUB binds a configurable localhost endpoint; the C++ SUB connects and subscribes to the `runtime_state` topic. PUB/SUB favors non-blocking latest-state delivery over per-message acknowledgement.
- **Serialization:** two ZeroMQ frames: ASCII topic and versioned JSON payload.
- **Thread model:** one C++ communication thread owns the SUB socket, validates messages, and publishes the latest state into process memory. ZeroMQ sockets are never shared across threads.
- **Synchronization:** the receiver replaces an immutable `RuntimeState` snapshot under a short mutex. Inference copies one snapshot at request start and uses it consistently for all exit decisions in that request.
- **Error handling:** malformed, out-of-range, duplicated, or out-of-order payloads are counted and ignored. Temporary disconnects retain the last state until TTL expiry.
- **Shutdown:** a stop token/atomic flag ends polling; the receiver thread is joined before the ZeroMQ context is destroyed.

No IPC occurs between neural-network layers.

## C++ Runtime and ONNX Runtime Integration

The runtime is C++17 and built with CMake. Phase 1 uses one `Ort::Session`, validates tensor names/shapes/types, executes a deterministic input, returns logits, and records warm-up-separated steady-state latency.

Standard ONNX Runtime cannot generally pause and resume an optimized graph. The multi-exit implementation will therefore use separately exported, benchmarkable stage models. A stage consumes either the original input or the preceding stage's feature tensor and produces both the next feature tensor and exit logits. The runtime owns stage sessions and invokes only as many as policy permits. This avoids unsupported graph mutation and makes per-stage cost measurable, at the cost of session boundaries and intermediate tensor movement.

ONNX Runtime APIs and allocator options will be checked against the installed version before use. No allocator or performance claim is made in the MVP.

## Multi-Exit Model Architecture

The target model is a modest ResNet-style Imagenette classifier:

```text
input -> stem -> block 1 -> exit 1
                     |-> block 2 -> exit 2
                                   |-> block 3 -> exit 3
                                                 |-> block 4 -> final exit
```

Training will use a weighted sum of cross-entropy losses. Each exit must be evaluated independently before C++ integration. Phase 1 intentionally uses a tiny fully connected model and synthetic deterministic input to validate the toolchain without conflating integration with model quality.

## Entropy, Confidence, and Adaptive Policy

For logits `z`, the runtime computes a numerically stable softmax and Shannon entropy `H = -sum(p_i * log(p_i))`. Confidence is `max(p_i)`. Unit tests will cover normalized, uniform, peaked, and extreme-logit cases.

The static baseline exits when entropy is below a calibrated threshold. The adaptive policy will map RuntimeState to an allowable stage budget and combine that budget with calibrated per-exit entropy thresholds. Thresholds will be selected on held-out data; they are not universal constants. Every decision will record its reason, considered threshold, state snapshot, and whether a deadline was met.

## Threading Model and Message Flow

The MVP runtime has a foreground inference/benchmark thread. The IPC phase adds exactly one receiver thread. The inference path never waits for a new control message. Later telemetry emission will be buffered away from timed inference regions.

```text
simulator PUB -> receiver thread -> validated latest-state snapshot
                                      |
request -> inference thread ----------+-> stage -> entropy/policy -> exit or next stage
                                                        |
                                                        v
                                                buffered metric record
```

## Benchmark Architecture

One reproducible harness will evaluate the same samples and preprocessing under three policies: full model, static early exit, and adaptive AeroNode. It will report accuracy, mean and P95 latency, CPU usage, peak resident memory, exit distribution, estimated compute saved, and deadline violations. Warm-up, repetition count, hardware/software versions, seeds, dataset split, thread counts, and raw result records will be preserved. Per-decision records will explain when and why an exit changed.

## Telemetry Architecture, FastAPI, and PostgreSQL

Telemetry is a supporting phase after the benchmark pipeline works. FastAPI will expose current state and recent inference measurements; it will not participate in inference decisions. PostgreSQL will store timestamped runtime states, inference measurements, benchmark runs, and configuration metadata. Writes will be asynchronous/batched so database latency is excluded from inference timing. The initial implementation will not create database objects.

## Build Architecture and Repository Structure

```text
AeroNode/
  README.md
  architecture.md
  current_status.md
  CMakeLists.txt
  model/
    training/
    export/
    artifacts/
  runtime/
    include/
    src/
    CMakeLists.txt
  tests/
  benchmarks/
    scripts/
    results/
  control_plane/
  api/
  docs/
```

Only files required by the current vertical slice are created. CMake discovers ONNX Runtime from an explicit `ONNXRUNTIME_ROOT` cache variable/environment path rather than assuming a global install.

## Component Responsibilities and Interfaces

- `model/training/simple_model.py`: deterministic Phase-1 PyTorch model and reference input.
- `model/export/export_simple_model.py`: ONNX export plus ONNX checker and Python Runtime comparison when dependencies are available.
- `runtime/src/main.cpp`: CLI model loading, inference, logits output, and latency summary.
- root/runtime CMake files: portable C++17 target and explicit ONNX Runtime dependency validation.
- Future `control_plane`: RuntimeState publisher and policy experiment inputs.
- Future `benchmarks`: dataset/policy orchestration and raw result aggregation.

The Phase-1 runtime CLI contract is `aeronode_runtime --model <path> [--iterations N]`; it emits machine-readable result fields after successful execution.

## Dependencies

Required for Phase 1: Python, PyTorch, ONNX, ONNX Runtime Python (verification), C++17 compiler, CMake, and ONNX Runtime C/C++ SDK. Later phases add torchvision/Imagenette, ZeroMQ/libzmq, a JSON parser, FastAPI, PostgreSQL, and a PostgreSQL Python driver.

Dependencies will be isolated/pinned after compatibility is verified; no global installation is assumed.

## Assumptions and Constraints

- Core inference remains C++.
- Control updates are slower than inference and latest-state semantics are sufficient.
- Initial execution is CPU-only on Windows x64; portability to Linux/WSL2 is preserved.
- Dynamic ONNX graph mutation is not an MVP mechanism.
- Dashboard, allocator experiments, and production hardening follow verified inference and benchmarks.
- Performance, fragmentation, memory determinism, and energy improvements are never claimed without measurements.

## Technical Decisions and Alternatives Considered

1. **Staged ONNX models, not pause/resume:** supported and measurable. A monolithic graph with arbitrary suspension is not promised by ONNX Runtime.
2. **PUB/SUB latest-state IPC:** avoids blocking inference. REQ/REP would add coupling; local UDP lacks ZeroMQ's socket semantics and is not selected because the implementation directive requires ZeroMQ.
3. **JSON first:** debuggable and adequate at low update frequency. Protobuf/FlatBuffers may reduce overhead but add premature schema/tooling complexity.
4. **Mutex-protected immutable snapshot:** simple and correct at 5 Hz. A lock-free structure is unnecessary until profiling demonstrates contention.
5. **GCC initially available:** MSYS2 GCC 15.2 is the only discovered compiler. The PDF recommends MSVC when available, but MSVC is not on `PATH`; CMake remains generator-neutral. This is an environment-driven deviation, not a permanent platform restriction.

## Known Limitations

- ZeroMQ is not installed and remains a prerequisite for the later IPC phase.
- The project-local Phase-1 dependencies and official ONNX Runtime 1.30.0 SDK are intentionally not committed; setup remains platform-specific until a bootstrap script is added.
- The official ONNX Runtime 1.30.0 Windows import library links and runs with MSYS2 GCC 15.2 for the current C-API-backed wrapper usage, but broader ABI compatibility will be rechecked as runtime features grow.
- Imagenette training cost, calibrated thresholds, stage-boundary overhead, and achievable accuracy are not known yet.
- Temperature is simulated unless a portable hardware sensor source is added later.
