# AeroNode

AeroNode is an experimental C++ edge-AI runtime that will select neural-network inference depth using model confidence and changing runtime constraints. The project focuses on a reproducible answer to one question: when can adaptive early exit reduce latency and compute while retaining acceptable accuracy?

The authoritative project brief is `AeroNode_Project_Brief (1).pdf`. See `architecture.md` for the actual design and `current_status.md` for what is implemented now.

## Current Scope

Phases 0–3 are complete. The active milestone is Phase 4:

```text
C++ ONNX Runtime multi-exit integration
  -> entropy-based early-exit policy
  -> ZeroMQ IPC (Python PUB / C++ SUB)
  -> per-exit latency and accuracy measurement
```

ZeroMQ control updates, adaptive policy, cross-policy benchmarks, allocator experiments, and telemetry remain subsequent milestones.

Phase 1 established the vertical slice: `PyTorch model → ONNX export → C++17 ONNX Runtime inference → Python/C++ output comparison → latency measurement`.

Phase 2 is now complete: the four-exit PyTorch model was trained on Imagenette and every classifier was independently validated. See `docs/phase2_validation.md` for the recorded metrics. Multi-exit ONNX/C++ integration remains intentionally unimplemented.

Phase 3 exports the accepted checkpoint as a dynamic-batch, four-output ONNX reference graph and verifies every exit against PyTorch. See `docs/phase3_validation.md`. C++ early-exit execution remains intentionally unimplemented.

## Repository Layout

- `model/training/` - model definitions and later training code.
- `model/export/` - ONNX export and verification tools.
- `model/artifacts/` - generated models and reference data (not source).
- `runtime/` - C++17 inference runtime.
- `tests/` - unit and integration tests.
- `benchmarks/` - benchmark scripts and generated results.
- `control_plane/` - later Python RuntimeState simulator.
- `api/` - later FastAPI telemetry service.

## Phase-1 Prerequisites

- Python with PyTorch, ONNX, ONNX Runtime, and NumPy.
- CMake 3.20 or newer.
- A C++17 compiler compatible with the chosen ONNX Runtime SDK.
- ONNX Runtime C/C++ SDK, exposed through `ONNXRUNTIME_ROOT`.

The verified direct Python/build versions are pinned in `requirements-phase1.txt`. Install them into a project-local virtual environment. The official ONNX Runtime 1.30.0 Windows x64 SDK was used for the verified run; set `ONNXRUNTIME_ROOT` to its extracted directory. No global dependencies were modified.

## Intended Phase-1 Workflow

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-phase1.txt
.\.venv\Scripts\python.exe model\export\export_simple_model.py --output model\artifacts\simple_model.onnx

$env:PATH = "C:\msys64\ucrt64\bin;C:\AeroNode\.venv\Scripts;" + $env:PATH
.\.venv\Scripts\cmake.exe -S . -B build -G Ninja `
  -DCMAKE_CXX_COMPILER=C:/msys64/ucrt64/bin/g++.exe `
  -DONNXRUNTIME_ROOT=C:/path/to/onnxruntime-win-x64-1.30.0
.\.venv\Scripts\cmake.exe --build build

.\.venv\Scripts\python.exe tests\verify_phase1.py `
  --executable build\runtime\aeronode_runtime.exe `
  --model model\artifacts\simple_model.onnx `
  --iterations 500
```

The verified reference run predicted class 1, matched Python/C++ logits within `1.4901161193847656e-08`, and measured 0.0035376 ms mean / 0.0037 ms P95 over 500 iterations. This is an integration measurement for a tiny synthetic model, not an application performance claim.

## Engineering Rules

- Core inference is C++17 and built with CMake.
- Python is used for training, export, simulation, orchestration, and telemetry—not core inference.
- RuntimeState IPC uses ZeroMQ and never sends a message per neural-network layer.
- ONNX Runtime behavior and APIs are verified against the installed version.
- Claims about latency, memory, compute, accuracy, or determinism require benchmark evidence.
- `current_status.md` is updated after each meaningful milestone.

## Phase-2 Reproduction

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-phase2.txt
.\.venv\Scripts\python.exe model\training\imagenette_data.py --data-root data
.\.venv\Scripts\python.exe -m unittest tests.test_multi_exit_model -v
.\.venv\Scripts\python.exe model\training\train_multi_exit.py `
  --data-root data `
  --output-dir model\artifacts\phase2-imagenette-5ep `
  --epochs 5 --batch-size 64 --workers 4
.\.venv\Scripts\python.exe model\training\evaluate_multi_exit.py `
  --checkpoint model\artifacts\phase2-imagenette-5ep\best_final_exit.pt `
  --data-root data `
  --output model\artifacts\phase2-imagenette-5ep\validation_report.json
```

## Phase-3 Reproduction

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_multi_exit_onnx -v
.\.venv\Scripts\python.exe model\export\export_multi_exit.py `
  --checkpoint model\artifacts\phase2-imagenette-5ep\best_final_exit.pt `
  --data-root data `
  --output model\artifacts\phase3\multi_exit_imagenette.onnx `
  --report model\artifacts\phase3\multi_exit_imagenette.verification.json `
  --verification-samples 10
```

## Generated Artifacts (Intentionally Excluded from Git)

`model/artifacts/` is excluded from version control because model checkpoints and ONNX exports are generated, deterministic artefacts that can be 100% reproduced from committed source code and the downloaded dataset. Including them would bloat the repository history with binary blobs that change on every training run and offer no meaningful diff.

The key Phase-3 artefact is:

| File | Size | SHA-256 |
|---|---|---|
| `model/artifacts/phase3/multi_exit_imagenette.onnx` | 10.70 MiB (11,217,740 bytes) | `0cc08e571383279798c00019f0079bcbc107db49a1fe8efca444a4b754f35051` |
| `model/artifacts/phase2-imagenette-5ep/best_final_exit.pt` | ~21.5 MiB | `32fb2975bcfe14d7311bc91db1553442391468d52c74f34a4efeb9d2aef27366` |

To regenerate these artefacts from scratch, follow the Phase-2 Reproduction workflow above (produces `best_final_exit.pt`), then run the Phase-3 Reproduction workflow above (produces `multi_exit_imagenette.onnx`).

After regeneration, verify SHA-256 integrity:

```powershell
(Get-FileHash model\artifacts\phase3\multi_exit_imagenette.onnx -Algorithm SHA256).Hash.ToLower()
# Expected: 0cc08e571383279798c00019f0079bcbc107db49a1fe8efca444a4b754f35051
```

The full numerical-parity validation record is in `docs/phase3_validation.md`.
