"""Fit entropy thresholds on a separate calibration subset of native outputs."""
import argparse
from datetime import datetime, timezone
import itertools
import math
import json
from pathlib import Path

import numpy as np

DEFAULT_GRID = (-1.0, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.3)


def split_samples(labels, seed):
    labels = np.asarray(labels)
    generator = np.random.default_rng(seed)
    calibration, evaluation = [], []
    for label in np.unique(labels):
        group = np.flatnonzero(labels == label)
        if len(group) < 2:
            raise ValueError("Each class needs at least two samples for disjoint fitting/evaluation")
        generator.shuffle(group)
        cut = len(group) // 2
        calibration.extend(group[:cut].tolist())
        evaluation.extend(group[cut:].tolist())
    return np.array(sorted(calibration)), np.array(sorted(evaluation))


def choose_exits(entropies, thresholds):
    entropies = np.asarray(entropies, dtype=np.float32)
    thresholds = np.asarray(thresholds, dtype=np.float32)
    if entropies.ndim != 2 or entropies.shape[1] != 4 or thresholds.shape != (4,):
        raise ValueError("Expected N x 4 entropy matrix and four thresholds")
    if not np.isfinite(entropies).all() or not np.isfinite(thresholds).all():
        raise ValueError("Entropy and thresholds must be finite")
    exits = np.full(len(entropies), 3, dtype=np.int64)
    for index in (2, 1, 0):
        exits[entropies[:, index] <= thresholds[index]] = index
    return exits


def fit(entropies, predictions, labels, costs, max_loss_pp=1.0, grid=DEFAULT_GRID):
    labels = np.asarray(labels)
    entropies = np.asarray(entropies, dtype=np.float32)
    predictions = np.asarray(predictions)
    costs = np.asarray(costs, dtype=np.float64)
    if len(labels) == 0 or predictions.shape != entropies.shape or entropies.shape != (len(labels), 4):
        raise ValueError("Invalid calibration tensor dimensions")
    if costs.shape != (4,) or not np.isfinite(costs).all() or (costs <= 0).any() or not math.isfinite(max_loss_pp) or max_loss_pp < 0:
        raise ValueError("Invalid cost vector or accuracy tolerance")
    if not np.isfinite(entropies).all() or (predictions < 0).any() or (predictions >= 10).any():
        raise ValueError("Invalid native prediction cache")
    if (labels < 0).any() or (labels >= 10).any() or not grid or not all(math.isfinite(value) for value in grid):
        raise ValueError("Invalid labels or threshold grid")
    baseline = float(np.mean(predictions[:, 3] == labels))
    rows, best = [], None
    for early in itertools.product(grid, repeat=3):
        thresholds = [float(np.float32(value)) for value in (*early, 2.5)]
        exits = choose_exits(entropies, thresholds)
        selected = predictions[np.arange(len(labels)), exits]
        accuracy = float(np.mean(selected == labels))
        cost = float(np.mean(costs[exits]))
        row = {"entropy_thresholds": thresholds, "accuracy_pct": accuracy * 100,
               "accuracy_loss_pp": (baseline - accuracy) * 100,
               "estimated_fixed_exit_cost_ms": cost, "mean_executed_stages": float(np.mean(exits+1))}
        rows.append(row)
        if (baseline - accuracy) * 100 <= max_loss_pp + 1e-10:
            key = (cost, -accuracy, tuple(thresholds))
            if best is None or key < best[0]: best = (key, row)
    if best is None:
        raise RuntimeError("No policy satisfies the calibration constraint, including full-depth fallback")
    return best[1], rows


def load_native_cache(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    count = manifest["selected_samples"]
    labels = np.array([row["label"] for row in manifest["images"]])
    entropies = np.full((count,4), np.nan, dtype=np.float32)
    predictions = np.full((count,4), -1, dtype=np.int64)
    with (directory / "records.jsonl").open(encoding="utf-8") as records:
        for line in records:
            row = json.loads(line)
            if row["repetition"] != 0 or not row["mode"].startswith("staged_exit_"): continue
            sample, exit = row["sample_index"], int(row["mode"][-1])-1
            if not 0 <= sample < count or row["label"] != labels[sample] or predictions[sample, exit] != -1:
                raise ValueError("Inconsistent/duplicate native cache record")
            if row["reference_max_abs_error"] > 1e-5:
                raise ValueError("Unverified native cache parity")
            entropies[sample, exit] = row["entropy"]
            predictions[sample, exit] = row["predicted_class"]
    if not np.isfinite(entropies).all() or (predictions < 0).any():
        raise ValueError("Native cache is incomplete")
    costs = [summary["modes"][f"staged_exit_{i}"]["mean_latency_ms"] for i in range(1,5)]
    return manifest, labels, entropies, predictions, costs


def calibrate(source, output, max_loss_pp=1.0, seed=20261013):
    output = Path(output)
    if output.exists() and any(output.iterdir()): raise FileExistsError("Existing calibration evidence is preserved")
    manifest, labels, entropy, predictions, costs = load_native_cache(source)
    calibration, evaluation = split_samples(labels, seed)
    best, candidates = fit(entropy[calibration], predictions[calibration], labels[calibration], costs, max_loss_pp)
    # Evaluation is reported only after selection; never used to retune the chosen thresholds.
    exits = choose_exits(entropy[evaluation], best["entropy_thresholds"])
    selected = predictions[evaluation, exits]
    baseline = float(np.mean(predictions[evaluation,3] == labels[evaluation]))
    accuracy = float(np.mean(selected == labels[evaluation]))
    output.mkdir(parents=True, exist_ok=True)
    policy = {"schema_version": 1, "entropy_thresholds": best["entropy_thresholds"],
              "calibration": {"created_utc": datetime.now(timezone.utc).isoformat(), "seed": seed,
                "samples": len(calibration), "objective": "Minimize mean fixed-exit measured-cost proxy subject to calibration accuracy constraint",
                "max_accuracy_loss_pp": max_loss_pp, "training_checkpoint_selection_used_validation": True}}
    (output / "policy.json").write_text(json.dumps(policy, indent=2), encoding="utf-8")
    result = {"calibration_samples": len(calibration), "evaluation_samples": len(evaluation),
              "chosen_on_calibration_only": best, "evaluation_accuracy_pct": accuracy*100,
              "evaluation_final_exit_accuracy_pct": baseline*100, "evaluation_accuracy_loss_pp": (baseline-accuracy)*100,
              "evaluation_target_met": (baseline-accuracy)*100 <= max_loss_pp,
              "calibration_sample_indices": calibration.tolist(), "evaluation_sample_indices": evaluation.tolist(),
              "candidate_count": len(candidates), "source": str(Path(source).resolve())}
    (output / "calibration.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (output / "candidates.json").write_text(json.dumps(candidates, indent=2), encoding="utf-8")
    for name, indices in (("calibration", calibration), ("evaluation", evaluation)):
        selection = {"dataset_indices": [manifest["images"][i]["dataset_index"] for i in indices],
                     "class_to_idx": manifest["class_to_idx"], "source_artifact_hashes": manifest["artifacts"]}
        (output / f"{name}_selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-accuracy-loss-pp", type=float, default=1.0)
    args = parser.parse_args()
    result = calibrate(args.source, args.output, args.max_accuracy_loss_pp)
    print(json.dumps({key:value for key,value in result.items() if not key.endswith("sample_indices")}, indent=2))
