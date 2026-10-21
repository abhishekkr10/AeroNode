# Evaluate AeroNode on another device

This experiment compares full-depth, static early-exit and adaptive inference under measured host state, with and without bounded study-owned CPU contention. It preserves the existing C++ policy; it does not fit a new timing model or guarantee deadlines.

## Prepare each Windows x64 device

Follow [quick start](quickstart.md) to build the runtime and verify the four stage exports. This benchmark additionally requires:

- The matching monolithic export at `model/artifacts/phase3/multi_exit_imagenette.onnx`.
- The same Imagenette dataset under `data/imagenette2-160`, or provide `--data-root`.
- The same repository revision and policy file on both devices.

Copy model/data artifacts separately from the source repository. Do not copy a build directory between devices. Run the command below sequentially on each machine, using distinct descriptive device labels and new result directories. Keep power settings, AC/battery state and unrelated workloads documented and reasonably consistent. Close optional workloads manually; the runner never stops unrelated processes or changes system settings.

## Run the protocol

```powershell
$env:PATH = 'C:\msys64\ucrt64\bin;' + $env:PATH
./.venv/Scripts/python.exe -m benchmarks.device_study run `
  --device laptop-a --output benchmarks/results/device-a `
  --samples 200 --repeats 3 --warmup 20 --rounds 2 --workers 2 `
  --duty 0.75 --deadline-ms 3 `
  --policy-config runtime/config/calibrated_imagenette_v1.json
```

On the second device use `--device laptop-b --output benchmarks/results/device-b` with the **same remaining options**. Use `--workers 1` on both devices if either cannot support two workers. The runner caps worker count at eight and normally reserves one logical processor. Workers run finite duty-cycle loops and are stopped in `finally`, including when a benchmark fails. Native execution has a default 120-second timeout per condition, configurable from 10 to 600 seconds. A worker expires after the native timeout plus 60 seconds; if preparation takes longer, the condition fails rather than being silently accepted.

The default study uses two rounds with alternating condition order: idle/contention, then contention/idle. Each case processes the same seeded, balanced sample selection using persistent native sessions, warmup, three repetitions and rotating mode order. Seven modes include the four forced-exit controls, which support numerical and execution-path verification. The report highlights full/static/adaptive results.

**Idle means no study-owned load workers.** It does not imply a quiet machine. Two 75%-duty workers are not a promise of 75% total host CPU usage; measured utilization is reported. No CPU affinity is imposed. Treat short CPU/memory observations as host state, not process-specific utilization or a causal isolation of every source of contention.

## Evidence

Each condition retains:

- `native/manifest.json`: hardware, data/model/source hashes, preprocessing, executable fingerprint and timing scope.
- `native/records.jsonl`: all seven modes, numerical parity, actual session counters, deadline outcomes and one shared observed state per paired image/repetition.
- `native/state.jsonl`: measured CPU/memory publisher audit; sequence/timestamp values match the native observations.
- `native/summary.json`: independently validated record matrix, accuracy, mean/P95 latency, exits and misses.
- `children.log`: worker/publisher diagnostics, including rejected sensor readings.

At the study root, `study.json` is a compact portable result and `report.md` presents each round separately. Existing outputs are never overwritten. Failed or interrupted studies retain evidence with `status: failed` and cannot be compared as completed results.

The host publisher samples at 20 Hz. A request pair obtains a fresh snapshot outside request timing, and all seven modes record it. Adaptive inference uses that snapshot; static/forced controls use generous internal budgets and zero pressure. All modes report misses against the pair's observed deadline. CPU and memory are measured, resource pressure is `max(CPU fraction,memory-used fraction)`, and temperature is a configured 60°C fallback. A transient invalid CPU reading is skipped without refreshing state; five consecutive invalid readings stop the publisher. Native inference rejects a stale snapshot rather than reverting silently during an experiment.

Timers exclude model loading, preprocessing, input file I/O, IPC snapshot acquisition, warmup and JSON serialization. They include tensor construction, validation, copies and prediction metrics. The full monolithic baseline returns all four heads, consistent with earlier studies. CPU and memory process summaries cover all resident sessions and are not per-mode resource-saving measurements.

## Compare devices

Transfer only the second machine's `study.json` to the first machine for the compact comparison; retain original evidence directories for audit.

```powershell
./.venv/Scripts/python.exe -m benchmarks.device_study compare `
  benchmarks/results/device-a/study.json C:/transferred/device-b/study.json `
  --output tmp/two-device-comparison.md
```

The comparison rejects incomplete/failed studies, duplicate condition identities, duplicate device labels, invalid metrics/parity and mismatched protocols, tensor/label hashes, model hashes, policy hashes or recorded source fingerprints. It does not authenticate a modified portable JSON file. Device labels are user supplied and do not prove two separate physical devices; review the hardware metadata.

Report accuracy, mean/P95 latency, misses, exit distributions and observed CPU/pressure. Keep rounds separate rather than averaging P95 values. A 200-image run is a pilot: it does not replace independent test-set validation or justify a broad accuracy claim. The 3-ms target is an explicit experimental choice, not a guarantee or a target inferred from the hardware.

## Next policy experiment

Use separate profiling runs to estimate stage/prefix costs before changing the fixed 3-ms estimate. Freeze those estimates before evaluation on different workloads and compare them against the current policy. This study deliberately preserves the current policy so it remains the baseline. New-device measurements and a separate evaluation are prerequisites for claiming a more reliable deadline policy.
