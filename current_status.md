# Current Status

## Completed

- Read and visually inspected all 8 pages of `AeroNode_Project_Brief (1).pdf`.
- Inspected the initial OS, CPU, compiler, build tools, Python packages, PostgreSQL service, VS Code extensions, and Git identity.
- Recorded the actual planned architecture, phase boundaries, interfaces, constraints, decisions, and known limitations in `architecture.md`.
- Established the minimum repository structure for model export, C++ runtime, tests, benchmarks, control plane, API, and documentation.
- Added a deterministic fixed-weight PyTorch model and ONNX export/check/parity script.
- Added a C++17 ONNX Runtime executable that validates tensor shapes, performs warm-up, emits logits/prediction, and measures mean/P95 latency.
- Added CMake configuration with explicit ONNX Runtime SDK discovery through `ONNXRUNTIME_ROOT`.
- Created an isolated `.venv` with PyTorch 2.14.0, ONNX 1.22.0, ONNX Runtime 1.30.0, NumPy 2.5.3, CMake 4.4.3, and Ninja 1.13.2.
- Exported and checked the Phase-1 ONNX model; Python ONNX Runtime matched PyTorch with maximum absolute error `1.4901161193847656e-08` and predicted class 1.
- Downloaded the official ONNX Runtime 1.30.0 Windows x64 C/C++ SDK into the ignored project-local `.deps` directory.
- Configured and built the C++17 runtime with CMake/Ninja and MSYS2 GCC 15.2.
- Completed the first vertical slice: C++ ONNX Runtime matched Python ONNX Runtime with maximum absolute error `1.4901161193847656e-08`, predicted class 1, and measured 500 iterations at 0.0035376 ms mean / 0.0037 ms P95 on the tiny synthetic model.
- Added an automated Phase-1 C++/Python parity verifier and pinned direct Phase-1 dependencies.
- Re-read `architecture.md` and `current_status.md` before Phase-2 implementation.
- Confirmed the installed PyTorch build is CPU-only, 14 CPU threads are available, and sufficient local storage exists.
- Selected the official Imagenette 160-pixel distribution and documented the concrete Phase-2 architecture, transforms, exit heads, and initial loss weights.
- Installed project-local torchvision 0.29.0 (matched to PyTorch 2.14.0) and Pillow 12.3.0.
- Implemented the compact four-stage ResNet-style model with four independent classifier heads.
- Implemented normalized weighted multi-exit cross-entropy and unit tests proving all four heads produce the expected shape and receive direct nonzero gradients.
- Implemented official Imagenette download/loading, reproducible augmentation, training, checkpointing, and per-exit loss/accuracy metric recording.
- Downloaded and validated the official Imagenette 160-pixel archive (`sha256:64d0c4859f35a461889e0147755a999a48b49bf38a7e0f9bd27003f10db02fe5`): 9,469 training images, 3,925 validation images, and 10 matching classes.
- Completed a bounded real-data training/evaluation smoke run, confirming data loading, four-exit metrics, backpropagation, optimizer updates, checkpointing, and JSON artifacts.
- Completed five full training epochs on all 9,469 Imagenette training images, recording aggregate and per-exit metrics after every epoch.
- Reloaded the best checkpoint in a separate validation-only process and evaluated all 3,925 validation images. Exit 1-4 accuracy was 57.27%, 66.73%, 71.80%, and 73.10%; loss was 1.3789, 1.0536, 0.8758, and 0.8300.
- Confirmed the validation-only report exactly reproduces the best in-training metrics, independently validating all four classifier heads.
- Re-read `architecture.md`, `current_status.md`, and `docs/phase2_validation.md` before Phase-3 implementation.
- Implemented the Phase-3 exporter using the accepted best epoch-5 checkpoint, four named logits outputs, dynamic batch dimensions, opset 17, embedded provenance metadata, structural checks, and per-exit parity checks.
- Added an automated temporary-model ONNX test that verifies four graph outputs, `[batch, 10]` shapes, dynamic batch execution, and numerical parity for every exit.
- Successfully exported the accepted epoch-5 checkpoint to `model/artifacts/phase3/multi_exit_imagenette.onnx` (11,217,740 bytes; SHA-256 `0cc08e571383279798c00019f0079bcbc107db49a1fe8efca444a4b754f35051`).
- Validated the graph with ONNX checker full checks and strict shape inference: opset 17, one dynamic-batch `[batch,3,128,128]` input, four `[batch,10]` outputs, 58 nodes, and 48 initializers.
- Verified PyTorch/ONNX Runtime parity on ten deterministic validation images spanning all ten classes. Maximum absolute error by exit was `2.3841858e-06`, `9.5367432e-07`, `1.4305115e-06`, and `1.6689301e-06`, within `atol=1e-5` and `rtol=1e-4`.
- Embedded source checkpoint hash/epoch, dataset hash, class mapping, loss weights, and image size in ONNX metadata.
- Completed the full Phase-3 regression suite: four tests passed, covering multi-exit shapes/supervision/loss validation plus ONNX four-output structure and parity.

## In Progress

- No implementation task is active; Phase 3 is complete and no C++ early-exit, ZeroMQ, or adaptive-policy work has started.

## Blocked

- ZeroMQ is not available yet; it is not needed until the IPC phase.

## Next Step

Begin Phase 4 by loading the monolithic four-output reference graph in C++ ONNX Runtime and verifying all logits against the Phase-3 reference; do not add adaptive early-exit behavior yet.

## Known Issues

- Global `cmake`, `ninja`, `make`, `pkg-config`, and `psql` are not on `PATH`; project-local CMake/Ninja are available.
- A PostgreSQL 18.6 Windows service is running; its client exists under `C:\Program Files\PostgreSQL\18\bin` but is not on `PATH`.
- Only MSYS2 GCC was discovered; MSVC `cl.exe` and Clang were not on `PATH`.
- PDF text extraction was unavailable through `pdftotext`; all pages were rendered and visually read instead.
- PyTorch 2.14 warns that the explicitly selected legacy TorchScript ONNX exporter is deprecated. It remains the verified Phase-3 path; migration to the dynamo exporter should be evaluated separately without changing this reference artifact.

## Technical Decisions

- Start with a tiny deterministic model and synthetic input to isolate integration correctness.
- Use separate ONNX stage models for the later multi-exit runtime rather than unsupported arbitrary pause/resume of one optimized graph.
- Use an explicit `ONNXRUNTIME_ROOT` for the C++ SDK.
- Keep Python outside the core inference path.
- Do not install or modify global dependencies during initial inspection.
- Keep generated models, downloaded SDKs, build output, and the virtual environment out of Git.
- Use the full official Imagenette train/validation splits with 128x128 crops for the initial CPU-only Phase-2 experiment.
- Use exit-loss weights `[0.15, 0.20, 0.25, 0.40]`; record them with run metrics and treat them as an initial experimental choice.
- Phase 3 first exports one four-output ONNX graph as the fidelity/reference artifact. Separate stage graphs remain the planned execution representation for later C++ early exit and are not part of this phase.
- Use ONNX opset 17 with dynamic batch dimensions, named outputs `exit_1_logits` through `exit_4_logits`, `atol=1e-5`, and `rtol=1e-4` for the accepted reference export.

## Environment / Dependencies

- OS: Microsoft Windows 11 Home Single Language, version 10.0.26200, 64-bit.
- CPU: Intel Core Ultra 5 225H, x64, 14 cores / 14 logical processors.
- C/C++: MSYS2 UCRT64 GCC/G++ 15.2.0 at `C:\msys64\ucrt64\bin\g++.exe`; MSVC and Clang not discovered on `PATH`.
- CMake: 4.4.3 in project-local `.venv`; not globally on `PATH`.
- Python: CPython 3.13.7.
- Project-local ML stack: PyTorch 2.14.0, torchvision 0.29.0, ONNX 1.22.0, ONNX Runtime 1.30.0, NumPy 2.5.3, and Pillow 12.3.0.
- Phase-2 vision dependencies: torchvision 0.29.0 and Pillow 12.3.0 in `.venv`.
- ONNX Runtime C++ SDK: official Windows x64 1.30.0, project-local under ignored `.deps`.
- ZeroMQ / pyzmq: not discovered.
- FastAPI: 0.141.1; Uvicorn 0.52.4.
- PostgreSQL: server/client 18.6; `postgresql-x64-18` service running; `psql` not on `PATH`; psycopg 3.3.4 and psycopg2-binary 2.9.12 installed globally.
- Testing: global pytest 9.1.1; project-local NumPy 2.5.3.
- VS Code: 1.137.0 x64 with Microsoft Python, C/C++, CMake Tools, and C++ Dev Tools extensions.
- Git: 2.53.0.windows.2; configured user identity exists; workspace initially not initialized as a repository.

## Recent Changes

- 2026-09-16: Completed authoritative PDF review and read-only environment inspection.
- 2026-09-16: Created initial architecture and status documentation.
- 2026-09-16: Added the minimum repository layout and Phase-1 Python/C++ source files.
- 2026-09-16: Provisioned isolated Python/build packages and completed PyTorch-to-ONNX/Python Runtime verification.
- 2026-09-16: Built and ran the C++17 ONNX Runtime executable and verified end-to-end output parity and latency measurement.
- 2026-09-16: Initialized Git and committed the verified Phase-1 vertical slice as `4e4ff02`.
- 2026-09-16: Started Phase 2 after re-reading architecture/status and documented the concrete multi-exit training design.
- 2026-09-16: Implemented and unit-tested the Phase-2 multi-exit model, weighted loss, Imagenette pipeline, and training/evaluation harness.
- 2026-09-16: Downloaded and validated the complete official Imagenette 160-pixel train/validation dataset.
- 2026-09-16: Completed the Phase-2 end-to-end smoke run and started full-split training.
- 2026-09-16: Completed five-epoch full-split training and independent checkpoint-reload validation of every exit.
- 2026-09-16: Committed the completed Phase-2 implementation and validation record as `240654c`.
- 2026-09-16: Started Phase 3 after re-reading all accepted architecture/status/validation documents.
- 2026-09-16: Implemented and unit-tested the four-output ONNX export/inspection/parity pipeline.
- 2026-09-16: Exported the accepted best checkpoint, passed full graph/shape validation, and verified all four exits against PyTorch on a class-diverse batch.
- 2026-09-16: Completed Phase 3 with four passing automated tests and a reproducible validation record.
- 2026-09-16: Committed the completed Phase-3 exporter, tests, and validation record as `4615a46`.
- 2026-09-16: Performed full repository audit; verified Phases 0-3 intact, all tests passing, no secrets in history. Renamed branch to `main`, configured `origin` remote to `https://github.com/abhishekkkr10/AeroNode.git`, hardened `.gitignore`, updated `README.md` and `current_status.md` for accuracy. Committed as `chore: prepare repository for public GitHub release`.
- 2026-09-17: Added "Generated Artifacts" section to `README.md` documenting explicit exclusion rationale for `multi_exit_imagenette.onnx` (~10.7 MiB) and `best_final_exit.pt` (~21.5 MiB): binary blobs are deterministically reproducible from source; size would bloat history. SHA-256 checksums and regeneration workflow recorded for future developers. Phase 2 + 3 test suite confirmed 4/4 passing. Pushed `main` to `https://github.com/abhishekkkr10/AeroNode.git`.
