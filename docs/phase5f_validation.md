# Phase 5F — threshold fitting and separate evaluation

Repository: https://github.com/abhishekkr10/AeroNode

Threshold fitting is complete for the accepted Imagenette checkpoint under the controlled normal RuntimeState. This phase preserves the Phase-4 decision rules. Configuration loading is opt-in; compiled defaults remain unchanged.

## Selection and results

- Native Phase-5E per-head entropy and predictions provide the fitting cache; Python performs analysis only.
- A deterministic class-stratified split (seed 20261013) uses 1,959 calibration images and 1,966 separate evaluation images. Sets are disjoint and together cover the 3,925 validation images.
- The declared engineering objective is to minimize a fixed-exit measured-cost proxy while losing at most one accuracy percentage point against the final head on the calibration subset. This is an initial engineering choice, not a universal optimal objective.
- The grid contains 2,197 threshold combinations. Selection uses calibration data only; evaluation results do not retune the chosen thresholds.
- Chosen thresholds in nats are `[0.8,0.8,1.4,2.5]`. The fourth head remains terminal independently of its threshold.
- Calibration accuracy is 72.2818%, with a 0.9699 percentage-point difference against its final-head baseline. Estimated prefix cost is explicitly a fitting proxy, not a measured calibrated-policy latency.

The frozen candidate was executed in C++ on all 1,966 evaluation images, with three repetitions and six modes: **35,388 measured requests**.

| Evaluation mode | Accuracy % | Mean latency ms | P95 ms |
|---|---:|---:|---:|
| Monolithic final-head baseline | 72.9400 | 2.8139 | 3.2859 |
| Frozen adaptive candidate | 72.6857 | 2.1052 | 3.0147 |

The evaluation accuracy difference is -0.2543 percentage points, meeting the declared normal-state constraint. Observed logit difference is 0 for every selected head, with argmax agreement. Exit counts across repetitions are `[474,1137,2625,1662]`. No request missed the normal-state 30-ms target in this run. These CPU measurements are specific to the recorded implementation/environment.

The evaluation split is held out from **threshold fitting**, but the underlying validation split was previously used for checkpoint selection during training. It is not an independent training test dataset. The one-percentage-point target is not guaranteed under forced deadlines or increased resource pressure.

## Configuration and reproduction

`runtime/config/calibrated_imagenette_v1.json` is the curated, validated configuration. It binds to checkpoint SHA-256 `32fb2975bcfe14d7311bc91db1553442391468d52c74f34a4efeb9d2aef27366`. The staged runtime rejects a supplied bound policy when the model checkpoint metadata does not match. Missing/malformed thresholds are rejected. Explicit threshold CSV overrides and a policy file are mutually exclusive.

```powershell
& .\.venv\Scripts\python.exe benchmarks/calibrate_policy.py `
  --source benchmarks/results/phase5e-full-20261012 --output benchmarks/results/new-calibration
& .\.venv\Scripts\python.exe benchmarks/run_phase5e.py `
  --selection benchmarks/results/new-calibration/evaluation_selection.json `
  --policy-config benchmarks/results/new-calibration/policy.json
& .\cmake-build-phase5e\runtime\aeronode_runtime.exe `
  --staged-dir model/artifacts/phase5 --input-bin tmp/phase5d/input_1.bin `
  --policy-config runtime/config/calibrated_imagenette_v1.json `
  --cpu-load 50 --temperature 60 --latency-budget 30 --resource-pressure 0.5
```

Generated fitting/cache evidence is under `benchmarks/results/phase5f-calibration-20261012/`; measured evaluation evidence is under `benchmarks/results/phase5f-evaluation-20261012/`. These directories remain ignored. Six tests cover disjoint/reproducible splitting, stage decisions, accuracy-constrained fitting, invalid matrices, configuration loading and checkpoint mismatch.
