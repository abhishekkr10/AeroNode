# Phase 9 — inference evidence and measured host state

Validated 2026-10-20 on Windows x64 CPU. This extension addresses an adoption gap: transferring the model, running an ordinary photo and inspecting why native inference stopped.

## Delivered

- `tools.evidence doctor`: stage integrity/order, label mapping, checkpoint-bound policy and executable fingerprint.
- `tools.evidence pack`: verified model-only transfer directory; refuses overwrite.
- `tools.evidence image`: benchmark-equivalent preprocessing; real C++ adaptive/full-depth execution; independent Python ONNX references for both outputs; execution-counter checks; immutable JSON and an offline HTML report.
- `control_plane.device`: measured Windows host CPU/memory; fixture-tested Linux parser; existing ZeroMQ publication and per-snapshot provenance audit. Pressure is a CPU/memory proxy; temperature is an explicit fallback.
- Checked Windows setup helper, artifact-free Windows/Linux CI configuration, quick start and an evidence-based interview walkthrough.

Core C++ inference, the Phase-4 policy, explicit-copy boundaries and IPC schema are unchanged.

## Validation

| Gate | Result |
|---|---|
| Clean Release rebuild | 16 actions; zero compiler warnings/errors |
| CTest | 6/6 entries including fixture setup; 71 C++ cases |
| Full Python discovery with real PostgreSQL | 73/73 cases |
| Subsequently added stale-IPC rejection | Passed in a 14/14 contract-case rerun |
| Distinct current cases exercised | **145: 71 C++ + 74 Python** |
| Phase-1 standalone parity | Class agreement; max error `1.4901161193847656e-08` |
| Phase-4 standalone parity | Batch 10, four exits; max error `5.960464477539063e-08` |
| Real PostgreSQL/HTTP upload/replay | 4,200 records; zero duplicate insertion on replay |
| Setup helper | Configure/build, 6/6 CTest and doctor completed |
| Final HTML | Browser-rendered and visually checked |

New tests exercise transfer integrity, overwrite protection, model tampering/missing files, mapping/path validation, policy mismatch, invalid deadlines, native failures without retry, counter validation, stale IPC rejection, exact benchmark preprocessing, real native image parity, tight-budget downstream skipping, CPU/reset math, Linux guest-time accounting, publisher validation and measured-state delivery into a native snapshot. Live integration matches the C++ sequence/timestamp and CPU value to the publisher audit.

Final local example: `ILSVRC2012_val_00009111.JPEG`, SHA-256 `35f017eff957abc92897ed441585a7e48076efe8213661b062d8e40a01f5483d`. Adaptive Exit 3 ran `[1,1,1,0]`; full-depth ran `[1,1,1,1]`. Both predicted tench; both independently checked logit errors were 0. This is one functional example, not an accuracy or speedup study.

Ignored evidence: `tmp/evidence-demo-20261020/release/{report.html,evidence.json}`, `tmp/transfer-20261020/phase5`, `benchmarks/results/phase6-scenarios-20261012/normal/telemetry_validation.json`, isolated PostgreSQL directory `.deps/telemetry-postgres-la0tbs17`. Integration helpers shut down their publisher/API/database processes.

## Reproduction

```powershell
$env:PATH = 'C:\msys64\ucrt64\bin;' + $env:PATH
./tools/setup.ps1 -OnnxRuntimeRoot C:/AeroNode_AG/.deps/onnxruntime-win-x64-1.30.0 -Test
$env:AERONODE_EVIDENCE_INTEGRATION = '1'
./.venv/Scripts/python.exe -m unittest tests.test_evidence tests.test_device -v
./.venv/Scripts/python.exe api/verify_telemetry.py `
  --directory benchmarks/results/phase6-scenarios-20261012/normal --full-regression
```

After the final added case, discovery contains 74 Python cases. Artifact-free CI selects `EvidenceContractTests` and `DeviceStateTests` (14 cases). Four new native/data-dependent cases require the opt-in environment and local artifacts. GitHub runner results must be checked separately after publication; local Windows execution does not prove Linux execution.

## Limits

No new inference speedup, energy saving or hard deadline guarantee is claimed. Demo process wall time includes startup/model loading/output. The full-depth control uses generous fixed resources, not the live adaptive snapshot. Existing training/evaluation limitations remain. Native Linux portability, new-device performance, thermal sensors and production API exposure are not validated. Hashes establish consistency with a supplied manifest, not publisher authenticity. The classifier has ten categories and no unknown-object rejection. Existing Python exporter and Starlette/httpx deprecation warnings remain; the zero-warning statement applies to the C++ compiler.
