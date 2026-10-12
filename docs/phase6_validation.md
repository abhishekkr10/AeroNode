# Phase 6 — full, static and adaptive policy scenarios

The benchmark now compares seven modes: monolithic full, four fixed staged prefixes, static entropy policy, and adaptive policy. The static policy uses the same frozen thresholds with a generous execution budget and no pressure forcing. The adaptive policy uses the existing confidence/deadline/pressure rules and the scenario snapshot.

Each scenario uses the same 200 deterministically selected class-balanced validation images, three repetitions, batch 1 and 20 warmup requests per mode. All **12,600 request records** preserve parity and execution-path counters. This bounded diagnostic sample does not represent full-split accuracy.

## Actual measured results

| Controlled scenario | Policy | Accuracy % | Mean ms | Deadline misses / 600 |
|---|---|---:|---:|---:|
| Normal: 30 ms, pressure 0.5 | Full | 77.0 | 2.7820 | 0 |
| Normal | Static | 77.5 | 1.9768 | 0 |
| Normal | Adaptive | 77.5 | 1.9961 | 0 |
| Strict deadline: 1 ms, pressure 0.5 | Full | 77.0 | 2.8338 | 600 |
| Strict deadline | Static | 77.5 | 2.0416 | 549 |
| Strict deadline | Adaptive | 67.5 | 0.9631 | 104 |
| Thermal-pressure input: 15 ms, pressure 0.95 | Full | 77.0 | 2.7586 | 0 |
| Thermal-pressure input | Static | 77.5 | 1.9615 | 0 |
| Thermal-pressure input | Adaptive | 78.0 | 1.7863 | 0 |

CPU load and temperature fields are **controlled RuntimeState inputs**, not measured sensor values or induced CPU/thermal load. The pressure case uses CPU-load input 92% and temperature input 85°C. Actual latency is measured on the workstation; no energy or thermal improvement is claimed.

In the deadline case, the adaptive path executes Exit 1 for all 600 requests. It reduces observed deadline misses, but still misses the 1-ms target for 104 requests. Therefore the existing stage estimate/decision heuristic provides no hard deadline guarantee. The accuracy fitting constraint applies to the normal state only; stricter resource constraints can sacrifice accuracy.

Static exit counts remain `[54,135,276,135]` across all three cases. Adaptive pressure-case counts are `[87,180,270,63]`, demonstrating a real execution-path change under the same frozen thresholds. The native test separately verifies that a strict budget stops adaptive execution while static execution continues to the final head.

For scenario comparisons, deadline outcomes use the common scenario target for every mode. A separate `policy_latency_budget_ms` field records the generous internal budget used to force full/static baselines. Mode order rotates through every enabled mode. The initial generated manifest's textual order description said six modes; the actual `modes` list and complete record matrix contain seven, and current source uses a mode-independent description.

```powershell
& .\.venv\Scripts\python.exe -m benchmarks.run_policy_scenarios `
  --output benchmarks/results/new-scenarios `
  --executable cmake-build-phase5e/runtime/aeronode_benchmark.exe `
  --policy-config runtime/config/calibrated_imagenette_v1.json
```

Full local evidence is retained in `benchmarks/results/phase6-scenarios-20261012/`. No policy defaults or ZeroMQ architecture were redesigned.
