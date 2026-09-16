"""Verify Phase 4 C++ ONNX Runtime multi-exit engine against Python ONNX Runtime."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import onnxruntime as ort
from PIL import Image
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

OUTPUT_NAMES = tuple(f"exit_{index}_logits" for index in range(1, 5))
ATOL = 1e-5
RTOL = 1e-4


def build_transform(image_size: int = 128) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def load_verification_batch(report_path: Path) -> tuple[np.ndarray, list[str]]:
    with report_path.open("r", encoding="utf-8") as f:
        report = json.load(f)

    verification_paths = report.get("verification_paths", [])
    if not verification_paths:
        raise RuntimeError("No verification_paths found in verification report")

    transform = build_transform(128)
    tensors = []
    for raw_path in verification_paths:
        # Convert path to project root relative if needed
        p = Path(raw_path)
        if not p.exists():
            # Try relative to PROJECT_ROOT
            alt_path = PROJECT_ROOT / p.relative_to(p.anchor)
            if not alt_path.exists():
                raise FileNotFoundError(f"Verification image not found: {raw_path}")
            p = alt_path
        with Image.open(p) as img:
            tensors.append(transform(img.convert("RGB")).numpy())

    batch = np.stack(tensors, axis=0).astype(np.float32)
    return batch, verification_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--executable",
        type=Path,
        default=PROJECT_ROOT / "build" / "runtime" / "aeronode_runtime.exe",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=PROJECT_ROOT / "model" / "artifacts" / "phase3" / "multi_exit_imagenette.onnx",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "model" / "artifacts" / "phase3" / "multi_exit_imagenette.verification.json",
    )
    parser.add_argument("--iterations", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    executable = args.executable.resolve()
    model_path = args.model.resolve()
    report_path = args.report.resolve()

    if not executable.exists():
        raise FileNotFoundError(f"Executable not found: {executable}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not report_path.exists():
        raise FileNotFoundError(f"Report not found: {report_path}")

    batch, paths = load_verification_batch(report_path)
    batch_size = batch.shape[0]

    tmp_dir = PROJECT_ROOT / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    input_bin = tmp_dir / "phase4_val_batch.bin"
    batch.tofile(input_bin)

    # 1. Run Python ONNX Runtime reference
    py_session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    py_outputs = py_session.run(list(OUTPUT_NAMES), {"input": batch})

    # 2. Run C++ executable
    completed = subprocess.run(
        [
            str(executable),
            "--model",
            str(model_path),
            "--input-bin",
            str(input_bin),
            "--batch-size",
            str(batch_size),
            "--iterations",
            str(args.iterations),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    cpp_result = json.loads(completed.stdout)

    # 3. Compare outputs
    exit_comparisons = []
    for exit_idx, (name, py_logits) in enumerate(zip(OUTPUT_NAMES, py_outputs)):
        cpp_exit = cpp_result["exits"][exit_idx]
        assert cpp_exit["name"] == name, f"Exit name mismatch: {cpp_exit['name']} vs {name}"

        # Reconstruct C++ logits tensor
        cpp_logits = np.zeros((batch_size, 10), dtype=np.float32)
        for pred in cpp_exit["predictions"]:
            sample_idx = pred["sample_index"]
            cpp_logits[sample_idx] = pred["logits"]

        max_err = float(np.max(np.abs(py_logits - cpp_logits)))
        parity = bool(np.allclose(py_logits, cpp_logits, atol=ATOL, rtol=RTOL))
        if not parity:
            raise RuntimeError(f"Numerical parity failed for {name}: max error {max_err}")

        py_preds = np.argmax(py_logits, axis=1)
        cpp_preds = [p["predicted_class"] for p in cpp_exit["predictions"]]
        if not np.array_equal(py_preds, cpp_preds):
            raise RuntimeError(f"Predicted classes differ for {name}")

        exit_comparisons.append(
            {
                "exit": name,
                "max_abs_error": max_err,
                "shape": list(cpp_logits.shape),
                "parity": parity,
            }
        )

    summary = {
        "status": "verified",
        "batch_size": batch_size,
        "iterations": args.iterations,
        "average_latency_ms": cpp_result["average_latency_ms"],
        "p95_latency_ms": cpp_result["p95_latency_ms"],
        "exit_comparisons": exit_comparisons,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
