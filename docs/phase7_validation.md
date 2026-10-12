# Phase 7 — allocator configuration experiments

The runner exposes two supported ONNX Runtime session options: CPU memory arena and memory-pattern planning. Both default to enabled, preserving prior behavior. The explicit vector-copy stage boundaries remain unchanged. No custom allocator, shared arena or zero-copy handoff was introduced.

Four combinations were measured in separate native processes. Each uses identical 200-image sampling, three repetitions, six modes, batch 1, one intra/inter-op thread and 20 warmup requests per mode. All **14,400 records** passed numerical parity and execution-path validation, including arena-disabled chaining.

| CPU arena | Memory pattern | Process peak working set MiB | Calibrated policy mean ms |
|---|---|---:|---:|
| Enabled | Enabled | 71.4883 | 1.9988 |
| Disabled | Enabled | 71.4453 | 2.0212 |
| Enabled | Disabled | 73.0195 | 1.9928 |
| Disabled | Disabled | 73.1680 | 2.0609 |

Peak memory covers the whole process with all five sessions resident, including runtime/DLL state. It is not per-stage or per-policy allocator usage. These small differences do not establish allocator superiority or fragmentation reduction. Defaults remain enabled because this bounded experiment does not justify a production default change.

The native regression case verifies all four option combinations and per-exit parity. Invalid allocator switch values are rejected. Standard `std::vector<float>` ownership and serialized request execution are retained.

```powershell
& .\.venv\Scripts\python.exe -m benchmarks.run_allocator_experiments `
  --output benchmarks/results/new-allocator-study `
  --executable cmake-build-phase5e/runtime/aeronode_benchmark.exe `
  --policy-config runtime/config/calibrated_imagenette_v1.json
```

Local raw evidence is in `benchmarks/results/phase7-allocator-20261012/`. Results are a Windows CPU configuration study, not a custom allocator or cross-platform determinism claim.
