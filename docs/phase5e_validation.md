# Phase 5E — CPU baseline measurement

Repository: https://github.com/abhishekkr10/AeroNode

Completed on 2026-10-12 using the existing Windows x64 CPU environment, ONNX Runtime 1.30.0, GCC 15.2, Release C++17, and one intra/inter-op thread. The benchmark uses persistent monolithic and staged sessions, batch 1, 20 untimed warmup requests per mode, all 3,925 Imagenette validation images, and three repetitions. Seed: 20261012.

## Measured results

| Mode | Accuracy % | Mean latency ms | Median ms | P95 ms |
|---|---:|---:|---:|---:|
| Monolithic full | 73.0955 | 3.0125 | 2.8273 | 3.9124 |
| Staged Exit 1 | 57.2739 | 1.0275 | 0.9434 | 1.4783 |
| Staged Exit 2 | 66.7261 | 1.6008 | 1.4722 | 2.2579 |
| Staged Exit 3 | 71.7962 | 2.2097 | 2.0449 | 3.0953 |
| Staged Exit 4 | 73.0955 | 3.0660 | 2.8787 | 4.0584 |
| Current adaptive policy | 72.8917 | 2.3953 | 2.2669 | 3.5752 |

There are 11,775 requests per mode and 70,650 total records. Every staged result matched its corresponding monolithic exit within absolute tolerance `1e-5`; observed maximum error was 0, with identical class predictions. The independently measured forced-exit accuracies reproduce the accepted per-head validation results.

The current adaptive policy selected Exits 1–4 for 534, 2,451, 3,720 and 5,070 requests respectively. Its measured mean latency was 20.49% lower than the monolithic baseline in this run, with an accuracy difference of -0.2038 percentage points. Full-depth staged execution was slightly slower than monolithic execution. These comparisons describe the current implementations on this CPU, including validation, copying and policy overhead; they do not isolate ORT compute cost or establish a universal hardware speedup.

The adaptive policy's unchanged entropy thresholds are `[0.60,0.80,1.10,2.50]`. It uses a controlled normal-state snapshot (CPU load 50%, temperature 60°C, pressure 0.5 and deadline 30 ms). These state values are inputs to the experiment, not sensor observations. No adaptive request missed its measured 30-ms budget in this run. Threshold calibration is a subsequent phase.

## Method and evidence

- Inputs use the existing validation preprocessing: resize 144, center crop 128, tensor conversion and ImageNet normalization.
- Every sample runs all six modes; mode order rotates and sample order is shuffled for each repetition. This reduces systematic order bias while preserving paired samples.
- Request timers include tensor construction, ORT calls, validations, explicit copies, prediction metrics and result ownership. Input file I/O, preprocessing, JSON recording, session creation and warmup are excluded.
- Whole-loop CPU time includes I/O and recording. Peak working set describes the whole process with all five sessions resident. Neither metric is a per-policy resource reduction claim.
- The baseline monolithic session returns all four outputs and computes all four heads' metrics. Stage policies execute their actual prefixes. Trace-derived execution counts are retained for every request.
- Mean latency by repetition is retained. Monolithic means were 2.8893, 2.8832 and 3.2650 ms, so environment variability remains visible rather than hidden by the aggregate.
- The manifest records hardware, active power scheme, image paths/hashes, class mapping, preprocessed input/label hashes, source/model/executable hashes, seed, mode order and measurement scope.

Local generated evidence is retained under `benchmarks/results/phase5e-full-20261012/`: `manifest.json`, `execution.json`, `records.jsonl`, `summary.json`, and `report.md`. These generated files and the 771,686,400-byte input matrix remain ignored by Git; this document and source reproduce the experiment.

## Validation and reproduction

Eight new Python cases cover summary accuracy/latency/count calculation, missing/duplicate paired requests, counter mismatches, invalid numerical data, label consistency, deterministic balanced sampling, real native execution, overwrite protection and bad binary files. All 111 cases pass: 71 C++ cases and 40 Python cases, plus both existing Phase-1/4 verification scripts. CTest passes 6/6 entries (including fixture setup). The clean Release build emits zero compiler errors or warnings. Release assertions remain active.

```powershell
$env:PATH = "C:\msys64\ucrt64\bin;" + $env:PATH
& .\.venv\Scripts\cmake.exe -S . -B cmake-build-phase5e -G Ninja `
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=C:/msys64/ucrt64/bin/g++.exe `
  -DCMAKE_MAKE_PROGRAM=C:/AeroNode_AG/.venv/Scripts/ninja.exe `
  -DONNXRUNTIME_ROOT=C:/AeroNode_AG/.deps/onnxruntime-win-x64-1.30.0 `
  -DPython3_EXECUTABLE=C:/AeroNode_AG/.venv/Scripts/python.exe
& .\.venv\Scripts\cmake.exe --build cmake-build-phase5e --clean-first --parallel 4
& .\.venv\Scripts\ctest.exe --test-dir cmake-build-phase5e --output-on-failure
$env:AERONODE_RUNTIME_EXECUTABLE = "$PWD\cmake-build-phase5e\runtime\aeronode_runtime.exe"
$env:AERONODE_BENCHMARK_EXECUTABLE = "$PWD\cmake-build-phase5e\runtime\aeronode_benchmark.exe"
& .\.venv\Scripts\python.exe -m unittest discover -s tests -p 'test_*.py' -v
& .\.venv\Scripts\python.exe benchmarks/run_phase5e.py
```

The default command creates a new timestamped result directory and processes the entire validation split. Existing results are never overwritten. `--samples 20 --repeats 1 --warmup 1` provides a small functional run without representing full-split accuracy.

Energy, FLOPs, hardware temperature, per-policy CPU usage, per-policy memory and deployment performance were not measured. The benchmark's accuracy evaluation uses the project's validation set, which was previously used for checkpoint selection; it is not an independent training test set.
