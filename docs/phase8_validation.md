# Phase 8 — local FastAPI and PostgreSQL telemetry

The local telemetry MVP is implemented and verified against **real PostgreSQL 18.6 and real HTTP**. It stores benchmark provenance, inference records and versioned RuntimeState snapshots. Inference remains entirely C++.

## Execution boundary

The C++ benchmark writes per-request JSONL after each timed request. This persisted file is the telemetry outbox. A separate uploader drains it in batches after inference completes; HTTP and database operations cannot delay the measured inference call. No per-layer IPC, database calls or Python inference were introduced.

Uploads derive a stable run UUID from the raw-record hash. Run metadata and inference keys are immutable/idempotent. Replaying the same data inserts zero additional rows; conflicting content returns HTTP 409 and rolls back the entire batch. Input validation rejects malformed execution paths, invalid ranges, non-finite values and boolean schema versions.

The API exposes:

- `GET /health`
- `PUT /v1/benchmark-runs/{run_id}`
- `POST /v1/benchmark-runs/{run_id}/inferences`
- `GET /v1/benchmark-runs`
- `GET /v1/benchmark-runs/{run_id}/inferences`
- `PUT /v1/runtime-states/{source_id}`
- `GET /v1/runtime-states/{source_id}`

`api/schema.sql` initializes three project-namespaced tables with primary keys, a run foreign key and a recent-record index. Queries use parameter binding. Connections are scoped to transactions and released after requests. Batch size is bounded at 1,000 records; metadata is limited to 2 MB and query limits are bounded.

## Verified results

Eight PostgreSQL integration cases pass: persisted queries/health, idempotent replay, atomic conflict rollback, unknown runs, invalid execution counters, RuntimeState validation/latest ordering, metadata conflicts/query bounds, and non-finite error handling. These cases use an isolated project-local cluster rather than the user's existing PostgreSQL service.

The real HTTP uploader persisted **all 35,388 calibrated evaluation records**, then replayed the complete file with **zero duplicate insertions**. PostgreSQL's stored row count matched the original raw record count. An earlier gate also persisted/replayed 4,200 seven-mode scenario records. The API and isolated database were shut down after verification; logs and database evidence remain under ignored `.deps/telemetry-postgres-*` directories.

Final regression validation passed **127 cases: 71 C++ and 56 Python**, plus both existing Phase-1/4 verification scripts. CTest passed 6/6 entries, including fixture preparation. The final clean Release build has zero compiler warnings/errors. Existing legacy-export deprecation warnings and the installed Starlette test-adapter deprecation warning remain visible in Python test output.

## Reproduction and local use

```powershell
& .\.venv\Scripts\python.exe -m pip install -r requirements-phase8.txt
& .\.venv\Scripts\python.exe api/verify_telemetry.py `
  --directory benchmarks/results/phase5f-evaluation-20261012 --full-regression
```

The verification helper discovers the installed PostgreSQL binaries, creates a fresh cluster only inside `.deps`, binds to loopback on an unused port, and stops its own processes. `AERONODE_PG_BIN` can point to another installed PostgreSQL binary directory. It does not change credentials or data in an existing service.

For an explicitly configured local database:

```powershell
$env:AERONODE_DATABASE_URL = 'postgresql://aeronode:YOUR_LOCAL_PASSWORD@127.0.0.1:5432/aeronode'
& .\.venv\Scripts\python.exe -m uvicorn api.server:app --host 127.0.0.1 --port 8080
# In a separate terminal, after a benchmark has completed:
& .\.venv\Scripts\python.exe -m api.upload_benchmark `
  --directory benchmarks/results/phase5f-evaluation-20261012 --url http://127.0.0.1:8080
```

No database URL is guessed and no real credentials are committed. The example file contains a placeholder. This is a loopback-only development service; public deployment, authentication, hardware sensor integration and operational production hardening are outside this MVP validation.
