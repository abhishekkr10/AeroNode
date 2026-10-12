"""Validate native paired benchmark records and compute auditable summaries."""
from collections import Counter, defaultdict
import json
import math
from pathlib import Path

import numpy as np

MODES = ("monolithic_full", "staged_exit_1", "staged_exit_2", "staged_exit_3", "staged_exit_4", "adaptive_current")
REASONS = {"FULL_DEPTH_BASELINE", "CONFIDENCE_MET", "LATENCY_BUDGET_EXHAUSTED",
           "RESOURCE_PRESSURE_FORCED_EXIT", "FINAL_EXIT_REACHED"}


def summarize(path: Path, samples: int, repeats: int, calibrated: bool = False, include_static: bool = False) -> dict:
    if samples <= 0 or repeats <= 0:
        raise ValueError("Samples and repeats must be positive")
    groups = defaultdict(list)
    modes = (*MODES[:5], "adaptive_calibrated") if calibrated else MODES
    if include_static: modes = (*modes,"static_calibrated" if calibrated else "static_current")
    seen = set()
    labels = {}
    with path.open(encoding="utf-8") as records:
        for line in records:
            row = json.loads(line)
            mode, sample, repetition = row["mode"], row["sample_index"], row["repetition"]
            if mode not in modes or not 0 <= sample < samples or not 0 <= repetition < repeats:
                raise ValueError("Unknown mode or out-of-range sample/repetition")
            key = (mode, sample, repetition)
            if key in seen:
                raise ValueError("Duplicate benchmark request")
            seen.add(key)
            if not 0 <= row["label"] < 10 or not 0 <= row["predicted_class"] < 10:
                raise ValueError("Invalid class index")
            if sample in labels and labels[sample] != row["label"]:
                raise ValueError("Inconsistent label for paired sample")
            labels[sample] = row["label"]
            latency, error = row["latency_ms"], row["reference_max_abs_error"]
            if not math.isfinite(latency) or latency <= 0 or not math.isfinite(error) or not 0 <= error <= 1e-5:
                raise ValueError("Invalid timing or numerical parity")
            if not math.isfinite(row["entropy"]) or not math.isfinite(row["confidence"]) or not 0 <= row["confidence"] <= 1:
                raise ValueError("Invalid prediction metrics")
            exit = row["selected_exit"]
            if not 1 <= exit <= 4 or row["reason"] not in REASONS:
                raise ValueError("Invalid exit or decision reason")
            expected_runs = [0] * 4 if mode == "monolithic_full" else [int(i < exit) for i in range(4)]
            if row["stage_run_counts"] != expected_runs or row["monolithic_session_runs"] != int(mode == "monolithic_full"):
                raise ValueError("Execution-path counter mismatch")
            if mode == "monolithic_full" and exit != 4:
                raise ValueError("Monolithic baseline must return Exit 4")
            if mode.startswith("staged_exit_") and exit != int(mode[-1]):
                raise ValueError("Forced baseline exited at wrong stage")
            if row["deadline_missed"] != (latency > row["latency_budget_ms"]):
                raise ValueError("Deadline outcome mismatch")
            groups[mode].append(row)
    if len(seen) != len(modes) * samples * repeats:
        raise ValueError("Incomplete benchmark record matrix")
    result = {}
    for mode in modes:
        rows = groups[mode]
        times = np.array([row["latency_ms"] for row in rows], dtype=np.float64)
        accuracy = lambda selected: 100.0 * sum(row["label"] == row["predicted_class"] for row in selected) / len(selected)
        per_repeat = [[row for row in rows if row["repetition"] == i] for i in range(repeats)]
        result[mode] = {
            "requests": len(rows), "unique_samples": samples, "accuracy_pct": accuracy(rows),
            "accuracy_pct_by_repetition": [accuracy(selected) for selected in per_repeat],
            "mean_latency_ms": float(times.mean()), "median_latency_ms": float(np.median(times)),
            "p95_latency_ms": float(np.percentile(times, 95, method="linear")),
            "minimum_latency_ms": float(times.min()), "maximum_latency_ms": float(times.max()),
            "mean_latency_ms_by_repetition": [float(np.mean([row["latency_ms"] for row in selected])) for selected in per_repeat],
            "exit_distribution": dict(Counter(str(row["selected_exit"]) for row in rows)),
            "decision_reasons": dict(Counter(row["reason"] for row in rows)),
            "total_stage_runs": [sum(row["stage_run_counts"][i] for row in rows) for i in range(4)],
            "deadline_misses": sum(row["deadline_missed"] for row in rows),
            "reference_max_abs_error": max(row["reference_max_abs_error"] for row in rows),
        }
    baseline = result["monolithic_full"]["mean_latency_ms"]
    for mode in modes:
        result[mode]["mean_latency_ratio_to_monolithic"] = result[mode]["mean_latency_ms"] / baseline
    return {"schema_version": 1, "measurement_type": "measured CPU benchmark", "batch_size": 1,
            "samples": samples, "repeats": repeats, "record_count": len(seen),
            "label_distribution": dict(Counter(str(label) for label in labels.values())),
            "percentile_method": "NumPy linear interpolation", "modes": result}
