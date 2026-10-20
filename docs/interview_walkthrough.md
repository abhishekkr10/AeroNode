# AeroNode: interview walkthrough

## The problem

An edge device may have limited CPU time and changing contention. Running every neural-network layer for every input wastes an opportunity: some inputs can be classified earlier, while uncertain inputs benefit from more computation. AeroNode makes that trade-off explicit and measurable.

## 60-second explanation

“I built a C++17 inference runtime for a four-exit image classifier. It executes separately exported ONNX stages, calculates prediction entropy and uses a deterministic policy to decide whether to stop. Stopping physically skips downstream sessions. A Python control plane publishes resource snapshots through ZeroMQ without blocking inference. I validated outputs against independent references, measured full/static/adaptive baselines, separated threshold fitting from evaluation, and stored evidence through an asynchronous replay workflow. The demo shows the decision trace and model fingerprints, so a prediction is explainable operationally.”

## Five-minute demonstration

1. Run `tools.evidence doctor`: explain model integrity, label mapping and checkpoint-bound configuration.
2. Run `tools.evidence image` on a category-appropriate photo. Open the report and inspect entropy, threshold, reason and actual session counters.
3. Repeat with a very tight budget in a new directory. Explain why earlier termination can sacrifice accuracy and still miss the deadline.
4. Start `control_plane.device` and repeat with IPC. Match the snapshot sequence to the audit; distinguish measured CPU/memory, proxy pressure and fallback temperature.
5. Present the existing persistent benchmark results with hardware, dataset and timing scope. Do not use cold demo process times to claim a speedup.

## Engineering decisions worth discussing

| Decision | Reason and trade-off |
|---|---|
| Four ONNX stages | Supported physical early termination; adds session/copy overhead compared with a single full graph. |
| Explicit owned feature copies | Clear tensor lifetimes and testable boundaries; potential bandwidth/allocation cost. |
| One resource snapshot per request | Consistent decisions and no per-layer IPC; changes during a request are observed next time. |
| Deterministic entropy policy | Inspectable, reproducible decisions; confidence can be miscalibrated and timing estimates are heuristic. |
| Artifact binding and independent parity | Catches mixed checkpoints and incorrect execution; hashes do not authenticate an untrusted publisher. |
| Persistent benchmark versus cold demo | Separates inference measurements from startup and demonstration costs. |
| Persisted telemetry outbox | HTTP/database failure cannot block inference; requires replay and idempotent storage. |
| Host sensing with provenance | Makes experiments less synthetic without pretending pressure proxies or fallback temperature are sensor facts. |

## Honest answers to difficult questions

- **Is it always faster?** No. Full-depth staged execution adds overhead; benefit depends on exit distribution and hardware. Cite only recorded paired measurements.
- **Does confidence mean correctness?** No. Softmax scores and low entropy can be confidently wrong. Reliability calibration and distribution-shift testing remain valuable extensions.
- **Is the evaluation independent?** Threshold fitting and evaluation subsets are disjoint, but the original validation dataset was used for checkpoint selection. A separate test dataset is still needed.
- **Is this a hard real-time scheduler?** No. The fixed stage-time estimate is heuristic; recorded tight-budget experiments include misses.
- **Does it measure energy or thermal improvement?** No. CPU/memory readings are available; pressure is derived and temperature is explicitly a fallback.
- **Is it production ready?** It is a validated experimental runtime with reproducibility tools. Cross-device validation, sustained-run evidence, sensor coverage and deployed-service hardening remain outstanding.

## Evidence-based resume wording

“Built a C++17/ONNX Runtime adaptive early-exit inference engine with physical stage skipping, numerical parity checks, ZeroMQ resource snapshots, reproducible policy experiments and PostgreSQL telemetry replay.”

For numerical bullets, use the exact dataset/hardware scope in `phase5e_validation.md` or `phase5f_validation.md`. Do not generalize the observed latency improvement to other devices or claim measured energy savings.
