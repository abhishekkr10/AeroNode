# Phase 5D — sequential staged C++ inference

Verified on 2026-10-12 on the existing Windows x64 CPU toolchain: GCC 15.2, C++17, CMake/Ninja, and ONNX Runtime 1.30.0. Phase 5D is complete for functional staged execution. This record contains no staged performance benchmark or speedup claim.

## Implementation and execution

`StagedInferenceEngine::Infer` extends the Phase-5C engine. Four sessions remain persistent. A request runs Stage 1, validates both outputs, computes per-sample prediction metrics, and calls `ExitDecisionEngine::EvaluateStage`. The incremental entry point uses the same rule implementation as the existing Phase-4 `Evaluate`, with the same defaults and rule priority. Only Exit 4 is inherently terminal in staged mode.

For each continuing boundary:

1. Validate float32, runtime shape, element count, and finite values.
2. Resize the appropriate owned vector to that validated count.
3. Copy the ORT output into the vector while the output is still alive.
4. Construct a new input tensor wrapping the vector.
5. Run the next session.

| Boundary | Explicit buffer | Shape | Elements |
|---|---|---|---|
| Stage 1 → 2 | `feat_1` | `[B,32,32,32]` | `B × 32768` |
| Stage 2 → 3 | `feat_2` | `[B,64,16,16]` | `B × 16384` |
| Stage 3 → 4 | `feat_3` | `[B,128,8,8]` | `B × 8192` |

Input is `[B,3,128,128]`; every exit returns `[B,10]`. Constructor checks validate the actual model contracts, dynamic batch dimension, stage identity, Phase-5A metadata, and consistent checkpoint provenance. Metadata introspection keeps the owning `Ort::TypeInfo` alive while using its unowned tensor-info view.

Returning an exit immediately returns a copied, owned logits vector. No later stage is invoked and no unused boundary copy is performed. Run counters are incremented at the actual `Ort::Session::Run` call site. Warmup is separately invoked, runs all four sessions, and is excluded from request counters.

Requests use one copied RuntimeState snapshot. `Infer` and `WarmUp` are serialized with a mutex. Batches use a common exit: the whole batch continues if any sample requests continuation. Individual samples are not compacted or retained at different earlier exits. The Phase-5C low-level accessors remain available, but require external synchronization and careful borrowed-tensor lifetimes; production callers should use `Infer`.

## Execution-path and numerical evidence

The tests use deterministic Imagenette validation images selected by the Phase-5B class-diverse helper, with the same 128-pixel preprocessing. Batches 1, 3, and 5 are checked at every forced exit. Test thresholds force paths only; they are not calibrated deployment values. Budget and pressure controls prevent incidental exits in forced-path checks.

| Selected exit | Actual session run counts (Stages 1–4) | Maximum absolute logit error |
|---|---|---|
| Exit 1 | `[1,0,0,0]` | `0` |
| Exit 2 | `[1,1,0,0]` | `0` |
| Exit 3 | `[1,1,1,0]` | `0` |
| Exit 4 | `[1,1,1,1]` | `0` |

These results hold for all three batch sizes against both the monolithic C++ session and independently generated Python monolithic ONNX reference outputs. All 36 sample/exit comparisons agree on argmax class (`(1+3+5) × 4`). The C++ tests enforce absolute error ≤ `1e-5` without a relative tolerance allowance. The existing Phase-5B staged/PyTorch suite also passes without changing its tolerance.

Tests additionally compare the actual copied buffers against independently run stage outputs and verify distinct storage addresses. Repeated requests with changing batch sizes preserve previously returned logits. Concurrent requests to the same engine succeed through serialization. A mixed-confidence batch proves that one sample permitting an exit does not prematurely stop other samples.

Five adversarial Stage-2 models trigger runtime feature/logit shape mismatch, non-finite feature/logit data, or an ONNX Runtime Gather failure. Each failing model is then left loaded and the same engine successfully executes Exit 1 with `[1,0,0,0]`. This independently demonstrates that the downstream session is skipped; the proof does not depend solely on the selected exit number.

## Errors and regression results

Negative tests cover empty/nonexistent/corrupt models; malformed tensor rank, shape, type, output count and dynamic batch metadata; incorrect stage ordering; missing/malformed stage properties; malformed or mixed checkpoint hashes; invalid input dimensions/counts; empty buffers; overflow; NaN/Inf; invalid RuntimeState; scratch/input aliasing; and actual ORT Run failures. Failed inference is not retried. A later independent valid request can still use the engine.

| Suite | Passing cases |
|---|---:|
| Existing C++ entropy | 5 |
| Existing C++ state receiver | 3 |
| Existing C++ Phase-4 decision | 4 |
| Existing C++ Phase-5C runner | 5 |
| New C++ Phase-5D execution | 54 |
| Python Phase-2 model / Phase-3 export | 4 |
| Python Phase-4 control plane / integration | 8 |
| Python Phase-5A export | 4 |
| Python Phase-5B parity | 5 |
| New Python staged CLI integration | 11 |
| Total unit/integration cases | **103** |

The 11 CLI cases include all 12 forced exit/batch combinations, manual state validation, fallback state, and live ZeroMQ snapshot reception. Legacy integration tests accept `AERONODE_RUNTIME_EXECUTABLE` so they validate the newly built executable rather than an older binary.

Two additional existing verification scripts pass: Phase 1 has maximum absolute error `1.4901161193847656e-08` and class 1 agreement; Phase 4 has maximum absolute error `5.960464477539063e-08` at each of four exits for batch 10. They were run with one iteration for functional regression; their existing timing fields are not used as Phase-5D benchmark evidence.

CTest passes **6/6 entries**: five C++ suites plus the Python fixture setup entry. That setup entry is not counted as an additional assertion case. Release assertions remain enabled for the older C++ suites. The clean Release build completes with zero compiler errors and zero compiler warnings under `-Wall -Wextra -Wpedantic`. No static-analysis gate is configured. The existing Python legacy-export deprecation warnings remain; the expected ORT error log from the deliberate failing model is part of the negative test.

## Reproduction

From the repository root, using the installed project-local tools:

```powershell
$env:PATH = "C:\msys64\ucrt64\bin;" + $env:PATH
& .\.venv\Scripts\cmake.exe -S . -B cmake-build-phase5d -G Ninja `
  -DCMAKE_BUILD_TYPE=Release `
  -DCMAKE_CXX_COMPILER=C:/msys64/ucrt64/bin/g++.exe `
  -DCMAKE_MAKE_PROGRAM=C:/AeroNode_AG/.venv/Scripts/ninja.exe `
  -DONNXRUNTIME_ROOT=C:/AeroNode_AG/.deps/onnxruntime-win-x64-1.30.0 `
  -DPython3_EXECUTABLE=C:/AeroNode_AG/.venv/Scripts/python.exe
& .\.venv\Scripts\cmake.exe --build cmake-build-phase5d --clean-first --parallel 4
& .\.venv\Scripts\ctest.exe --test-dir cmake-build-phase5d --output-on-failure
$env:AERONODE_RUNTIME_EXECUTABLE = "$PWD\cmake-build-phase5d\runtime\aeronode_runtime.exe"
& .\.venv\Scripts\python.exe -m unittest discover -s tests -p 'test_*.py' -v
& .\.venv\Scripts\python.exe tests/verify_phase1.py `
  --executable cmake-build-phase5d/runtime/aeronode_runtime.exe `
  --model model/artifacts/simple_model.onnx --iterations 1
& .\.venv\Scripts\python.exe tests/verify_phase4_engine.py `
  --executable cmake-build-phase5d/runtime/aeronode_runtime.exe --iterations 1
```

For one staged request, use the generated validation input:

```powershell
& .\cmake-build-phase5d\runtime\aeronode_runtime.exe `
  --staged-dir model/artifacts/phase5 --input-bin tmp/phase5d/input_1.bin `
  --batch-size 1 --latency-budget 1000000 --resource-pressure 0 `
  --entropy-thresholds 3,-1,-1,-1
```

The command returns Exit 1 logits, shape `[1,10]`, run counts `[1,0,0,0]`, and a decision trace. Staged mode executes one request and requires preprocessed float32 input; it emits no latency benchmark. `--ipc-endpoint` uses the existing StateReceiver. The original `--model` mode remains available for Phase-1/4 regression.

Fixtures are generated under ignored `tmp/phase5d`; ONNX artifacts, checkpoints, dataset, SDK, and build outputs remain excluded from Git. Existing Phase-5A/B exports and tests were preserved.

## Remaining limits

- Functional validation covers the installed Windows x64 CPU environment; other platforms/providers were not tested.
- The available MinGW toolchain has no AddressSanitizer runtime. Shape/count checks, buffer content/address checks, ownership checks, repeated batch resizing and concurrency tests passed; this is not a sanitizer-certified memory audit.
- Existing thresholds and stage estimates are unchanged and uncalibrated. Hardware performance, allocator work and telemetry remain Phase 5E+.
- Checkpoint metadata validates consistency, not artifact authentication. Export/hash integrity remains covered by the existing Phase-5A suite.
