# Phase 10 — device-study tooling and local validation

Validated 2026-10-21. **The experiment tooling and local study are complete; an actual second-device run remains pending.** This record must not be presented as cross-device validation or completion of a new deadline policy.

## Implementation

- Persistent C++ benchmark accepts `--ipc-endpoint`. It requires a fresh initial state and rejects stale state during execution.
- One snapshot is acquired outside request timers for each paired image/repetition. All modes record that observation; adaptive inference uses it. Static/forced controls retain generous internal budgets. All seven modes report deadline outcomes against the same observed target when static comparison is enabled.
- Core inference, Phase-4 rule priority/fixed timing estimate and explicit-copy tensor boundaries are unchanged. The benchmark now links the existing ZeroMQ receiver.
- `benchmarks.device_study` runs alternating idle/contention rounds, bounded study-owned CPU workers, real host-state publication, source/model/data fingerprints, audit matching and compact portable reports.
- Worker count, duty cycle and duration are bounded. Cleanup terminates only owned children, including after failures. Native timeouts preserve partial evidence; failed studies cannot enter the comparison.
- Comparison rejects mismatched protocols, recorded source fingerprints, data/model/policy hashes, incomplete/duplicate conditions, duplicate labels, invalid metrics and failed parity. A portable JSON file is not cryptographically authenticated; labels do not prove separate physical machines.
- Transient invalid host readings are skipped without refreshing native state. Five consecutive invalid readings stop publication. Tests cover both behaviors.

## Local experiment

Host: Windows 11 x64, Intel Core Ultra 5 225H, 14 logical processors as reported by CIM, 16,505,851,904 physical memory bytes, Balanced power scheme, ONNX Runtime 1.30.0, Release GCC 15.2 C++17. This is the existing development device.

Protocol: 200 seeded balanced Imagenette validation images; batch 1; three repetitions; 20 warmup requests per mode; two rounds with reversed condition order; seven modes; two contention workers at requested duty 0.75; controlled 3-ms deadline; frozen existing calibrated policy. **16,800 measured requests**, all selected-head logit errors 0 with class agreement against the monolithic reference. Each mode has 600 requests per condition.

| Round | Condition | Host CPU mean % | Full P95 ms | Adaptive P95 ms | Full misses / 600 | Adaptive misses / 600 |
|---|---|---:|---:|---:|---:|---:|
| 1 | Idle | 24.57 | 10.1831 | 4.0524 | 448 | 187 |
| 1 | Contention | 45.33 | 9.7703 | 3.8302 | 549 | 267 |
| 2 | Contention | 28.85 | 10.0154 | 3.9849 | 464 | 321 |
| 2 | Idle | 24.84 | 10.1928 | 3.9695 | 470 | 175 |

Full-depth accuracy was 70%, static calibrated accuracy 71%, and adaptive calibrated accuracy 59% in every condition. This **11-percentage-point adaptive accuracy reduction** under the tight budget is material. Fewer deadline misses and lower P95 do not establish an acceptable deployment policy. The existing calibration target applied to normal controlled state, not this tight-budget/live-state study.

Mean pressure proxy ranged from 0.833 to 0.942 across cases, so memory pressure was substantial even without study-owned CPU workers. Conditions also varied across rounds. There is no claim that workers isolated every causal effect or achieved a fixed total CPU utilization. The same previously checkpoint-selected validation data is reused; this is not independent test-set accuracy. No energy/thermal measurement is made.

All native paired snapshots matched publisher audit identities/values. Three isolated sensor-reading rejections occurred across the completed study and were skipped; no stale state was accepted. An earlier attempt stopped when the original publisher exited on one invalid counter reading. That attempt is retained with `status: failed` at `benchmarks/results/phase10-local-20261021` and excluded from the results above.

## Gates

| Gate | Actual result |
|---|---|
| Clean Release build | 16 build actions; zero compiler errors/warnings |
| CTest | 6/6 entries, including fixture setup; 71 C++ cases |
| Full Python discovery with native integration enabled | 85/85 cases |
| Total distinct C++/Python cases | **156** |
| Standalone Phase-1 / Phase-4 verifiers | Passed; max errors `1.4901161193847656e-08` / `5.960464477539063e-08` |
| Real PostgreSQL/HTTP replay of a new live-state case | 4,200 records uploaded; zero duplicate insertion on replay |
| Artifact-free CI configuration | 23 contract cases on Windows/Linux; remote execution is not asserted here |

Existing Python exporter and Starlette/httpx deprecation warnings remain. No new sanitizer/static-analysis claim is made. The benchmark executable used in the tests is the newly built `cmake-build-phase5e/runtime/aeronode_benchmark.exe`.

## Reproduction and retained evidence

See [device-study guide](device_study.md) for the exact two-machine commands and timing boundaries. Local run:

```powershell
$env:PATH = 'C:\msys64\ucrt64\bin;' + $env:PATH
./.venv/Scripts/python.exe -m benchmarks.device_study run `
  --device development-windows --output benchmarks/results/new-device-study `
  --samples 200 --repeats 3 --warmup 20 --rounds 2 --workers 2 `
  --duty 0.75 --deadline-ms 3 --policy-config runtime/config/calibrated_imagenette_v1.json
$env:AERONODE_EVIDENCE_INTEGRATION = '1'
./.venv/Scripts/python.exe api/verify_telemetry.py `
  --directory benchmarks/results/phase10-local-20261021-validated/round-1-idle/native --full-regression
```

Raw results, summaries, manifests and audits remain ignored under `benchmarks/results/phase10-local-20261021-validated`. The current compact table was regenerated from completed `study.json` into ignored `tmp/phase10-local-report.md`, adding means and exit distributions without rerunning or changing measurements. Real database evidence is in `.deps/telemetry-postgres-2mhidc_2`; the validation helper shut down its API/database. Workload/publisher children were shut down after each condition.

Next required external step: execute the same protocol on a second Windows x64 device and compare its `study.json`. Separate timing-profile fitting/evaluation is a subsequent policy experiment; the fixed 3-ms heuristic remains the baseline.
