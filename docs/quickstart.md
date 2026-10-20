# Run AeroNode on a photo

The supported execution environment is currently Windows x64 CPU. The native runtime remains C++17/ONNX Runtime; Python only preprocesses the image, invokes the executable and independently verifies the resulting logits. There is no Python substitute for native adaptive inference.

## 1. Build once

Clone **https://github.com/abhishekkr10/AeroNode**. Use Python 3.13, the MSYS2 UCRT64 GCC/ZeroMQ/nlohmann JSON dependencies and the extracted Windows x64 ONNX Runtime 1.30.0 SDK. The helper checks prerequisite paths, installs pinned Python dependencies into `.venv` and builds Release binaries:

```powershell
./tools/setup.ps1 -OnnxRuntimeRoot C:/path/to/onnxruntime-win-x64-1.30.0
```

The script does not install the system compiler or download an SDK. Existing CMake caches must use the same compiler/generator. GCC's DLL directory must remain on PATH when running the executable:

```powershell
$env:PATH = 'C:\msys64\ucrt64\bin;' + $env:PATH
```

## 2. Transfer models without retraining

Models are deliberately absent from Git. On a machine with the accepted exports:

```powershell
./.venv/Scripts/python.exe -m tools.evidence pack --output tmp/transfer/phase5
```

Transfer that entire directory into `model/artifacts/phase5` on the target. It contains four stage files and their export manifest, not executable binaries, datasets or checkpoints. Build native binaries on the target. The image demo does **not** need the Imagenette dataset or training checkpoint. To export a different trained checkpoint, use `model/export/export_staged.py --help`; recalibrate its policy separately.

```powershell
./.venv/Scripts/python.exe -m tools.evidence doctor `
  --policy-config runtime/config/calibrated_imagenette_v1.json
```

Doctor verifies all stage-file hashes, stage order, label mapping, policy/checkpoint binding and executable presence. It records platform and artifact fingerprints. This is integrity against the accompanying manifest, **not cryptographic authentication**; obtain bundles from a trusted source. Native execution subsequently validates graph metadata and loadability. Doctor alone does not prove DLL availability.

## 3. Generate an inference evidence report

```powershell
./.venv/Scripts/python.exe -m tools.evidence image --image C:/photos/example.jpg `
  --policy-config runtime/config/calibrated_imagenette_v1.json `
  --output tmp/demo-normal
```

Open `tmp/demo-normal/report.html` in a browser. The offline report displays adaptive/full-depth predictions, executed/skipped stages, entropy thresholds, reasons and verification scope. Adjacent `evidence.json` preserves raw native traces and provenance. New directories are required; previous evidence is never overwritten. Original photos are not copied into reports. File and tensor hashes are retained.

To demonstrate deadline pressure, choose a new output directory and add `--deadline-ms 0.001`. This is a policy demonstration, not a promise that the request meets that deadline. The full-depth control uses a generous budget to reach stage 4.

The report independently checks both native predictions against Python ONNX Runtime staged execution with absolute logit tolerance `1e-5` and class agreement. It checks native session counters to prove actual skipping. Preprocessing matches the validation benchmark: RGB conversion, resize 144, center crop 128, tensor conversion and ImageNet normalization. EXIF rotation is not applied.

## 4. Use actual host CPU and memory

In terminal A:

```powershell
./.venv/Scripts/python.exe -m control_plane.device --audit tmp/device-session.jsonl --count 600
```

In terminal B, while the publisher is running:

```powershell
./.venv/Scripts/python.exe -m tools.evidence image --image C:/photos/example.jpg `
  --ipc-endpoint tcp://127.0.0.1:5555 --output tmp/demo-device
```

The publisher samples Windows host CPU counter deltas and physical memory availability. Linux `/proc` parsing is also implemented and fixture-tested; Linux native inference remains unvalidated and still has Windows-specific build dependencies. Resource pressure is a **proxy**: `max(CPU fraction, memory-used fraction)`. Temperature defaults to a configured 60°C fallback and the deadline is a configured target. The audit records each field's provenance. No thermal or power measurement is claimed. Windows CPU counters may be limited by processor-group behavior on very large hosts.

The image command rejects a stale IPC fallback. Its saved sequence/timestamp can be matched to the publisher's JSONL audit. The IPC snapshot overrides CLI deadline values; the report displays the actual received budget. The normal-state calibrated policy has not been revalidated across all live states.

## 5. Measure performance separately

Cold process wall times in the report include startup, session loading, inference and JSON output. They are not comparable inference benchmarks. Use persistent sessions, a dataset, warmup and repeated paired requests:

```powershell
./.venv/Scripts/python.exe benchmarks/run_phase5e.py --samples 20 --repeats 1 --warmup 1
```

Then omit the sample/repeat/warmup overrides for the documented full evaluation. See `phase5e_validation.md` for timing boundaries and `phase5f_validation.md` for calibration/evaluation separation. No latency gain, energy saving or accuracy claim follows from one photo.

## Troubleshooting

| Failure | Action |
|---|---|
| Model or manifest missing | Transfer the complete stage bundle; cloning Git is insufficient. |
| Integrity mismatch | Obtain the original exports; do not edit the manifest to bypass verification. |
| Policy/checkpoint mismatch | Use matching artifacts or omit the opt-in policy; recalibrate a new model independently. |
| Native runtime fails before JSON output | Verify ORT and MSYS2 DLL availability and matching x64 architecture. |
| No fresh IPC state | Start the publisher first, check endpoint, then retry with a new output directory. |
| Incorrect answer on an unrelated object | The prototype recognizes ten categories and has no unknown-object rejection. |

For native tests, the existing fixture setup additionally requires the local checkpoint and Imagenette data. API/PostgreSQL dependencies are only needed for telemetry validation, not image inference.
