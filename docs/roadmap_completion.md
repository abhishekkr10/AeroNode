# AeroNode MVP roadmap completion

The original MVP remains complete through Phase 8. A subsequent Phase-9 usability/host-sensing extension is recorded in [Phase 9 validation](phase9_validation.md). It adds photo-based native evidence, verified artifact transfer and host CPU/memory publication without changing the original policy or claiming new benchmark gains.

The [Phase-10 device-study extension](phase10_validation.md) implements and locally validates live-state persistent benchmarks and bounded contention experiments. A real second-device comparison remains pending; the original exit policy is preserved.

Publishing destination: **https://github.com/abhishekkr10/AeroNode**.

The user-authorized documented MVP roadmap is implemented and validated through Phase 8. This records completed engineering work and its actual limits; it does not claim every possible research extension is complete.

| Milestone | Evidence |
|---|---|
| 1–4: training/export, reference runtime, entropy, policy and ZeroMQ | Existing Phase-1/2/3/4 validation records and regression suites |
| 5A–5C: stage export, parity, persistent sessions/buffers | Existing export, Python parity and C++ session tests |
| 5D: physical staged execution/skipping | [Phase 5D](phase5d_validation.md) |
| 5E: full-validation CPU baseline | [Phase 5E](phase5e_validation.md) |
| 5F: threshold fitting and separate evaluation | [Phase 5F](phase5f_validation.md) |
| 6: full/static/adaptive controlled scenarios | [Phase 6](phase6_validation.md) |
| 7: ORT allocator configuration study | [Phase 7](phase7_validation.md) |
| 8: FastAPI/PostgreSQL telemetry and replay | [Phase 8](phase8_validation.md) |

Measured experiments contain 133,038 native requests: 70,650 full-validation baseline requests, 35,388 separate-evaluation requests, 12,600 controlled-scenario requests and 14,400 allocator-study requests. Every record retains numerical parity and actual session execution counters.

The frozen normal-state policy meets the declared one-percentage-point accuracy-loss target on the separate threshold-evaluation subset. Under a 1-ms controlled deadline, early exit still misses some deadlines and sacrifices accuracy. The allocator study does not justify changing default arena settings. Telemetry stores and replays the measured records through real PostgreSQL/HTTP, without entering the C++ inference path.

Final checks: 127 C++/Python cases, two existing standalone parity verifiers, 6/6 CTest entries and a clean warning-free Release build. Generated datasets/models/results/SDKs/build outputs and real credentials stay excluded from Git. The curated model-bound policy and all reproducible source/validation documentation are included.

Remaining boundaries are explicit: Windows CPU validation only; RuntimeState pressure/temperature scenarios are controlled inputs; validation data was previously used for checkpoint selection; no energy/FLOP or hardware thermal measurement; no custom allocator/zero-copy claim; and no public authenticated production telemetry deployment. These are extensions beyond the delivered experimental MVP, not silently completed features.
