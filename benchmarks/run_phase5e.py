"""Prepare reproducible image inputs, invoke C++ inference, and preserve evidence."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks.summarize_phase5e import summarize
from model.training.imagenette_data import create_datasets


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def select_indices(targets, count, seed):
    if count == 0:
        count = len(targets)
    if count < 1 or count > len(targets):
        raise ValueError("Sample count exceeds validation set or is negative")
    groups = defaultdict(list)
    for index, label in enumerate(targets):
        groups[label].append(index)
    rng = random.Random(seed)
    for group in groups.values():
        rng.shuffle(group)
    classes = sorted(groups)
    rng.shuffle(classes)
    selected = []
    while len(selected) < count:
        for label in classes:
            if groups[label] and len(selected) < count:
                selected.append(groups[label].pop())
    return selected


def hardware_metadata():
    result = {"platform": platform.platform(), "machine": platform.machine(),
              "logical_processors": os.cpu_count(), "python": platform.python_version()}
    if os.name == "nt":
        command = ("$c=Get-CimInstance Win32_Processor; $s=Get-CimInstance Win32_ComputerSystem; "
                   "@{cpu_name=$c.Name;physical_cores=$c.NumberOfCores;logical_processors=$c.NumberOfLogicalProcessors;"
                   "physical_memory_bytes=$s.TotalPhysicalMemory}|ConvertTo-Json -Compress")
        completed = subprocess.run(["powershell.exe", "-NoProfile", "-Command", command],
                                   capture_output=True, text=True, check=True)
        result.update(json.loads(completed.stdout))
        power = subprocess.run(["powercfg.exe", "/GETACTIVESCHEME"], capture_output=True, text=True)
        result["power_scheme"] = power.stdout.strip() if power.returncode == 0 else "unavailable"
    return result


def run(output_dir, executable, samples=0, repeats=3, warmup=20, seed=20261012, data_root=None):
    if repeats <= 0 or warmup <= 0 or seed <= 0:
        raise ValueError("Repeats, warmup, and seed must be positive")
    if sys.byteorder != "little":
        raise RuntimeError("The current binary benchmark contract requires a little-endian host")
    executable = Path(executable).resolve()
    if not executable.is_file():
        raise FileNotFoundError(executable)
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("Choose an empty or new result directory; existing results are preserved")
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    _, validation = create_datasets(Path(data_root or ROOT / "data/imagenette2-160"), 128)
    indices = select_indices(validation.targets, samples, seed)
    manifest = []
    input_path, labels_path = output_dir / "inputs.bin", output_dir / "labels.bin"
    with input_path.open("wb") as destination:
        for index in indices:
            image, label = validation[index]
            image.numpy().astype("<f4").tofile(destination)
            image_path = Path(validation.samples[index][0])
            manifest.append({"dataset_index": index, "path": str(image_path.relative_to(validation.root)),
                             "label": label, "image_sha256": sha256(image_path)})
    np.array([entry["label"] for entry in manifest], dtype="<i4").tofile(labels_path)
    stage_dir = ROOT / "model/artifacts/phase5"
    model = ROOT / "model/artifacts/phase3/multi_exit_imagenette.onnx"
    artifacts = [model, *[stage_dir / f"stage_{i}.onnx" for i in range(1, 5)]]
    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "hardware": hardware_metadata(),
        "validation_set_size": len(validation), "selected_samples": len(indices),
        "class_to_idx": validation.class_to_idx, "sampling_seed": seed, "images": manifest,
        "input_sha256": sha256(input_path), "labels_sha256": sha256(labels_path),
        "artifacts": {str(path.relative_to(ROOT)): sha256(path) for path in artifacts},
        "executable_sha256": sha256(executable),
        "sources": {path: sha256(ROOT / path) for path in (
            "benchmarks/staged_benchmark.cpp", "benchmarks/run_phase5e.py", "benchmarks/summarize_phase5e.py",
            "runtime/src/staged_model_runner.cpp", "runtime/include/staged_model_runner.hpp", "runtime/include/exit_decision.hpp")},
        "preprocessing": "Validation Resize(144), CenterCrop(128), ToTensor, ImageNet mean/std normalization",
        "input_format": "little-endian float32 NCHW; native requests use batch 1",
        "timing_scope": "C++ synchronous request including tensor creation, validation, feature/logit copies and prediction metrics; excludes input file I/O, JSON output, preprocessing, session creation and warmup",
        "order": "Seeded sample shuffle each repetition; rotate the six modes for every paired sample",
        "cpu_scope": "Whole measurement loop including file I/O and JSON recording; percent relative to one logical core",
        "memory_scope": "Whole benchmark process peak working set, including all resident stage and reference sessions; not per-mode allocation usage",
        "state_scope": "Controlled normal-state snapshot, not hardware sensor telemetry",
        "calibration": "Current policy defaults are uncalibrated; no threshold fitting is performed",
    }
    (output_dir / "manifest.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    command = [str(executable), "--input-bin", str(input_path), "--labels-bin", str(labels_path),
               "--stage-dir", str(stage_dir), "--monolithic-model", str(model),
               "--records", str(output_dir / "records.jsonl"), "--samples", str(len(indices)),
               "--repeats", str(repeats), "--warmup", str(warmup), "--seed", str(seed)]
    environment = os.environ.copy()
    if os.name == "nt":
        environment["PATH"] = "C:\\msys64\\ucrt64\\bin;" + environment.get("PATH", "")
    completed = subprocess.run(command, cwd=ROOT, env=environment, stdout=subprocess.PIPE,
                               text=True)  # C++ progress is visible on stderr while it runs.
    if completed.returncode != 0:
        raise RuntimeError(f"Native benchmark failed with exit {completed.returncode}; partial evidence retained in {output_dir}")
    execution = json.loads(completed.stdout)
    if not execution.get("completed"):
        raise RuntimeError("Native benchmark did not complete")
    (output_dir / "execution.json").write_text(json.dumps(execution, indent=2), encoding="utf-8")
    summary = summarize(output_dir / "records.jsonl", len(indices), repeats)
    summary["execution"] = execution
    summary["hardware"] = provenance["hardware"]
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = ["# Phase 5E — measured CPU benchmark", "", f"Samples: {len(indices)} / {len(validation)} validation images; repetitions: {repeats}; batch: 1.",
             "", "| Mode | Accuracy % | Mean ms | Median ms | P95 ms | Exit counts 1–4 |", "|---|---:|---:|---:|---:|---|"]
    for mode, entry in summary["modes"].items():
        exits = ", ".join(str(entry["exit_distribution"].get(str(i), 0)) for i in range(1, 5))
        lines.append(f"| {mode} | {entry['accuracy_pct']:.4f} | {entry['mean_latency_ms']:.4f} | {entry['median_latency_ms']:.4f} | {entry['p95_latency_ms']:.4f} | {exits} |")
    lines += ["", "The manifest records hardware, hashes, preprocessing, mode order and timing scope. Raw per-request evidence is in records.jsonl.",
              "", "These measurements describe this Windows CPU run and current uncalibrated policy. CPU time and peak memory cover the entire process, including all resident models and recording overhead. No energy, thermal, or FLOP measurement is made."]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, default=ROOT / "cmake-build-phase5e/runtime/aeronode_benchmark.exe")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "benchmarks/results" / datetime.now(timezone.utc).strftime("phase5e-%Y%m%dT%H%M%S%fZ"))
    parser.add_argument("--samples", type=int, default=0, help="0 uses the full validation split")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20261012)
    args = parser.parse_args()
    result = run(args.output_dir, args.executable, args.samples, args.repeats, args.warmup, args.seed)
    print(json.dumps({"report": str(args.output_dir.resolve() / "report.md"), "records": result["record_count"],
                      "modes": result["modes"]}, indent=2))


if __name__ == "__main__":
    main()
