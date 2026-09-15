"""Export and verify the deterministic Phase-1 model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import onnx
import onnxruntime as ort
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.training.simple_model import REFERENCE_INPUT, create_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "model" / "artifacts" / "simple_model.onnx",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(0)
    model = create_model()
    with torch.inference_mode():
        torch_logits = model(REFERENCE_INPUT).cpu().numpy()

    torch.onnx.export(
        model,
        (REFERENCE_INPUT,),
        output_path,
        input_names=["input"],
        output_names=["logits"],
        opset_version=17,
        dynamo=False,
    )

    checked_model = onnx.load(output_path)
    onnx.checker.check_model(checked_model)

    session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    ort_logits = session.run(["logits"], {"input": REFERENCE_INPUT.numpy()})[0]
    max_abs_error = float(np.max(np.abs(torch_logits - ort_logits)))
    if not np.allclose(torch_logits, ort_logits, rtol=1e-5, atol=1e-6):
        raise RuntimeError(f"PyTorch/ONNX Runtime mismatch: max error {max_abs_error}")

    result = {
        "model": str(output_path),
        "input_shape": list(REFERENCE_INPUT.shape),
        "output_shape": list(ort_logits.shape),
        "logits": ort_logits.flatten().tolist(),
        "predicted_class": int(np.argmax(ort_logits, axis=1)[0]),
        "max_abs_error": max_abs_error,
        "opset": 17,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
