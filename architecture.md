# AeroNode Architecture

## Project Purpose

AeroNode is an experimental edge-AI inference runtime that changes neural-network inference depth in response to current resource constraints and model confidence. Its result must be a reproducible measurement of the accuracy, latency, and compute trade-off, not merely an early-exit model demonstration.

## Problem Statement and Core Engineering Question

Fixed-depth inference ignores changes in CPU load, thermal pressure, and latency requirements. AeroNode asks whether an adaptive early-exit policy can reduce edge-class latency and computational work while retaining acceptable accuracy under changing runtime constraints.

## Implementation State

The experimental MVP is implemented and verified through Phase 8. Phases 1–4 established training, the reference graph/runtime, entropy, policy and ZeroMQ snapshots. Phases 5A–5D implement verified staged execution; 5E measures CPU baselines; 5F fits and separately evaluates an opt-in model-bound threshold configuration. Phase 6 compares full/static/adaptive controlled scenarios; Phase 7 measures supported ORT allocator options; Phase 8 stores/replays telemetry through local FastAPI/PostgreSQL. `docs/roadmap_completion.md` links the evidence and explicit limits. `current_status.md` records the current state.

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

Standard ONNX Runtime cannot generally pause and resume an optimized graph. The runtime uses four separately exported stage models. A stage consumes the original input or the preceding stage's feature tensor and produces exit logits plus the next feature tensor (except Stage 4, which only returns logits). `StagedInferenceEngine::Infer` invokes each persistent session, validates tensor type/shape/count and finite values, then evaluates the existing Phase-4 policy. Returning at an intermediate exit prevents any downstream session invocation.

At each continuing boundary, the engine copies the ORT-owned feature tensor into `feat_1`, `feat_2`, or `feat_3`, then constructs a fresh input `Ort::Value` wrapping that vector. No session-owned `Ort::Value` crosses the boundary. A request mutex serializes `Infer` and `WarmUp`; results own their logits and traces. The retained Phase-5C raw session/buffer APIs require external synchronization and callers must not resize a buffer while a borrowed tensor is alive.

For a batch, every sample receives the existing policy evaluation at the current stage. The batch returns one common exit only when all samples permit exiting there; otherwise the entire batch continues. Phase 5D does not compact batches or freeze individual samples at different exits. One RuntimeState snapshot is used for the request. Elapsed time is used only by the existing deadline rule, with its unchanged stage estimate; no staged timing benchmark is emitted.

Phase 3 intentionally adds a distinct reference representation before staged execution: one opset-17 ONNX graph with input `input` shaped `[batch_size, 3, 128, 128]` and outputs `exit_1_logits` through `exit_4_logits`, each shaped `[batch_size, 10]`. It proves checkpoint provenance, graph validity, output preservation, dynamic-batch behavior, and PyTorch/ONNX numerical fidelity. It is suitable for the next C++ phase's full-graph multi-output baseline, but it cannot save compute merely by requesting an early output because ONNX Runtime may still execute the shared graph. Separate stage graphs remain the planned later execution representation. This clarifies two artifact roles rather than replacing the staged architecture.

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

Phase 2 uses the official 160-pixel Imagenette distribution and trains from random initialization. The concrete backbone is a compact ResNet-18-style network with four residual stages, two basic blocks per stage, channel widths `[32, 64, 128, 256]`, and 2,803,720 trainable parameters. Each stage ends in an independent head consisting of adaptive global average pooling and a linear 10-class classifier. All four logits tensors are returned together during PyTorch training/evaluation. The initial weighted loss is `0.15, 0.20, 0.25, 0.40` from earliest to final exit; the final head retains the largest influence while intermediate heads receive direct supervision. These weights are configuration recorded with every run, not claimed to be optimal.

Training uses random resized crops and horizontal flips at 128x128, while validation uses deterministic resize/center crop. The lower crop size is an explicit CPU-development trade-off while still using the complete Imagenette train/validation splits. Each epoch records weighted aggregate loss plus unweighted cross-entropy loss, top-1 accuracy, and sample count for every exit. A checkpoint is valid for later phases only after every head has been evaluated on the Imagenette validation split and the metrics artifact identifies the dataset classes, split sizes, seed, architecture, and loss weights.

The accepted Phase-2 prototype was trained from scratch for five epochs with SGD, momentum 0.9, initial learning rate 0.05, cosine decay, weight decay `5e-4`, batch size 64, and seed 20260916. A fresh reload of the best checkpoint evaluated all 3,925 validation images and reproduced per-exit accuracy of 57.27%, 66.73%, 71.80%, and 73.10%. This establishes that every classifier is useful and that accuracy generally improves with depth; it is not a claim that five epochs or the loss weights are optimal.

## Entropy, Confidence, and Adaptive Policy

For logits `z`, the runtime computes a numerically stable softmax and Shannon entropy `H = -sum(p_i * log(p_i))`. Confidence is `max(p_i)`. Unit tests will cover normalized, uniform, peaked, and extreme-logit cases.

The static baseline uses entropy thresholds with generous resources. The adaptive runtime preserves the Phase-4 confidence/deadline/resource-pressure rules, including its existing stage estimate. An optional model-bound configuration contains thresholds selected solely on a calibration subset, then evaluated separately. The compiled defaults remain unchanged. Every benchmark decision records the selected exit/reason, state scope, execution counters and deadline outcome. The heuristic does not guarantee hard deadlines; the controlled 1-ms experiment records actual misses.

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

The local FastAPI service exposes benchmark runs, recent inference measurements and source-scoped RuntimeState records. PostgreSQL stores provenance and timestamped records in project-namespaced tables initialized from `api/schema.sql`. C++ writes a persisted JSONL outbox outside request timers; a separate uploader batches these records after inference. Database/HTTP work does not participate in decisions or measured request latency. Idempotent keys support replay, while conflicting content rolls back the batch. Real integration validation uses isolated project-local PostgreSQL clusters, leaving existing services/data unchanged.

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
- `model/export/export_multi_exit.py`: best-checkpoint four-output ONNX export, provenance metadata, graph inspection, and PyTorch/ONNX Runtime parity report.
- `model/training/multi_exit_resnet.py`: compact four-stage ResNet backbone, exit heads, and weighted loss.
- `model/training/imagenette_data.py`: official Imagenette download, checksum, transforms, and split validation.
- `model/training/train_multi_exit.py`: reproducible training, per-exit metrics, and checkpoint selection.
- `model/training/evaluate_multi_exit.py`: checkpoint-reload validation of every exit.
- `runtime/src/main.cpp`: CLI model loading, inference, logits output, and latency summary.
- root/runtime CMake files: portable C++17 target and explicit ONNX Runtime dependency validation.
- `control_plane`: existing RuntimeState publisher; C++ receiver owns its socket and snapshots.
- `benchmarks`: native measurements, deterministic data preparation, threshold fitting, scenario/allocator studies and validated aggregation.
- `api`: local FastAPI/PostgreSQL storage, validated batched replay and isolated real-service validation.

The Phase-1 runtime CLI contract is `aeronode_runtime --model <path> [--iterations N]`; it emits machine-readable result fields after successful execution.

## Dependencies

Required for Phase 1: Python, PyTorch, ONNX, ONNX Runtime Python (verification), C++17 compiler, CMake, and ONNX Runtime C/C++ SDK. Phase 2 adds torchvision and the official Imagenette 160-pixel dataset. Later phases add ZeroMQ/libzmq, a JSON parser, FastAPI, PostgreSQL, and a PostgreSQL Python driver.

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
6. **Monolithic reference graph before stage graphs:** Phase 3 preserves all four outputs in one graph to establish a single fidelity baseline. It is not treated as an early-termination mechanism; later stage exports must be checked against both this graph and PyTorch.

## Known Limitations

- ZeroMQ is configured and verified in Phase 4 and in the Phase-5D staged CLI integration test; its local binaries remain an environment prerequisite.
- The project-local Phase-1 dependencies and official ONNX Runtime 1.30.0 SDK are intentionally not committed; setup remains platform-specific until a bootstrap script is added.
- The official ONNX Runtime 1.30.0 Windows import library links and runs with MSYS2 GCC 15.2 for the current C-API-backed wrapper usage, but broader ABI compatibility will be rechecked as runtime features grow.
- Five CPU epochs establish prototype behavior but are not a converged accuracy study; loss-weight/epoch calibration remains future experimental work.
- Compiled entropy thresholds remain unchanged; a curated, model-bound configuration is calibrated for the controlled normal state. CPU comparisons include validation, copying and policy costs. Other hardware, energy, physical thermal effects and isolated stage-boundary overhead remain unmeasured.
- The Phase-3 monolithic ONNX graph exposes all exits but does not itself guarantee compute savings when an early output is requested.
- Netron is not installed; ONNX checker, strict shape inference, operator inventory, metadata inspection, and ONNX Runtime execution provide the recorded reproducible inspection.
- Temperature is simulated unless a portable hardware sensor source is added later.
