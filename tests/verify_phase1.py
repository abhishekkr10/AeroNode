"""Run the C++ Phase-1 executable and verify it against Python ONNX Runtime."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
import onnxruntime as ort


REFERENCE_INPUT = np.array([[-1.0, 0.5, 2.0, -0.25]], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    completed = subprocess.run(
        [
            str(args.executable.resolve()),
            "--model",
            str(args.model.resolve()),
            "--iterations",
            str(args.iterations),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    cpp_result = json.loads(completed.stdout)

    session = ort.InferenceSession(
        str(args.model.resolve()), providers=["CPUExecutionProvider"]
    )
    python_logits = session.run(["logits"], {"input": REFERENCE_INPUT})[0].flatten()
    cpp_logits = np.asarray(cpp_result["logits"], dtype=np.float32)
    max_abs_error = float(np.max(np.abs(python_logits - cpp_logits)))
    if not np.allclose(python_logits, cpp_logits, rtol=1e-5, atol=1e-6):
        raise RuntimeError(f"C++/Python output mismatch: max error {max_abs_error}")
    if cpp_result["predicted_class"] != int(np.argmax(python_logits)):
        raise RuntimeError("C++ and Python predicted classes differ")

    print(
        json.dumps(
            {
                "status": "verified",
                "max_abs_error": max_abs_error,
                "predicted_class": cpp_result["predicted_class"],
                "average_latency_ms": cpp_result["average_latency_ms"],
                "p95_latency_ms": cpp_result["p95_latency_ms"],
                "iterations": cpp_result["iterations"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
