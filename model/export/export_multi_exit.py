"""Export the validated Phase-2 checkpoint as a four-output ONNX graph."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np
import onnx
from onnx import shape_inference
import onnxruntime as ort
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.training.imagenette_data import create_datasets, download_imagenette
from model.training.multi_exit_resnet import MultiExitResNet


INPUT_NAME = "input"
OUTPUT_NAMES = tuple(f"exit_{index}_logits" for index in range(1, 5))
OPSET_VERSION = 17
ABSOLUTE_TOLERANCE = 1e-5
RELATIVE_TOLERANCE = 1e-4


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def export_four_output_model(model: nn.Module, inputs: torch.Tensor, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    torch.onnx.export(
        model,
        (inputs,),
        output_path,
        input_names=[INPUT_NAME],
        output_names=list(OUTPUT_NAMES),
        dynamic_axes={
            INPUT_NAME: {0: "batch_size"},
            **{name: {0: "batch_size"} for name in OUTPUT_NAMES},
        },
        opset_version=OPSET_VERSION,
        do_constant_folding=True,
        dynamo=False,
    )


def value_shape(value_info: onnx.ValueInfoProto) -> list[int | str]:
    dimensions: list[int | str] = []
    for dimension in value_info.type.tensor_type.shape.dim:
        if dimension.HasField("dim_value"):
            dimensions.append(dimension.dim_value)
        elif dimension.HasField("dim_param"):
            dimensions.append(dimension.dim_param)
        else:
            dimensions.append("unknown")
    return dimensions


def validate_graph(model_path: Path, class_count: int = 10) -> dict[str, Any]:
    graph_model = onnx.load(model_path)
    onnx.checker.check_model(graph_model, full_check=True)
    inferred_model = shape_inference.infer_shapes(graph_model, strict_mode=True, data_prop=True)
    inputs = {value.name: value_shape(value) for value in inferred_model.graph.input}
    outputs = {value.name: value_shape(value) for value in inferred_model.graph.output}
    if list(inputs) != [INPUT_NAME]:
        raise RuntimeError(f"Unexpected ONNX inputs: {inputs}")
    if list(outputs) != list(OUTPUT_NAMES):
        raise RuntimeError(f"Unexpected ONNX outputs: {outputs}")
    if inputs[INPUT_NAME] != ["batch_size", 3, 128, 128]:
        raise RuntimeError(f"Unexpected input shape: {inputs[INPUT_NAME]}")
    for output_name, output_shape in outputs.items():
        if output_shape != ["batch_size", class_count]:
            raise RuntimeError(f"Unexpected shape for {output_name}: {output_shape}")
    operator_counts = Counter(node.op_type for node in inferred_model.graph.node)
    return {
        "ir_version": inferred_model.ir_version,
        "opset_imports": {
            item.domain or "ai.onnx": item.version for item in inferred_model.opset_import
        },
        "inputs": inputs,
        "outputs": outputs,
        "node_count": len(inferred_model.graph.node),
        "initializer_count": len(inferred_model.graph.initializer),
        "operator_counts": dict(sorted(operator_counts.items())),
    }


def compare_outputs(
    model: nn.Module,
    model_path: Path,
    inputs: torch.Tensor,
    absolute_tolerance: float = ABSOLUTE_TOLERANCE,
    relative_tolerance: float = RELATIVE_TOLERANCE,
) -> list[dict[str, Any]]:
    model.eval()
    with torch.inference_mode():
        torch_outputs = tuple(output.cpu().numpy() for output in model(inputs))
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    ort_outputs = session.run(list(OUTPUT_NAMES), {INPUT_NAME: inputs.cpu().numpy()})
    if len(torch_outputs) != 4 or len(ort_outputs) != 4:
        raise RuntimeError("Expected four outputs from both PyTorch and ONNX Runtime")

    comparisons: list[dict[str, Any]] = []
    for output_name, torch_output, ort_output in zip(
        OUTPUT_NAMES, torch_outputs, ort_outputs, strict=True
    ):
        maximum_error = float(np.max(np.abs(torch_output - ort_output)))
        parity = bool(
            np.allclose(
                torch_output,
                ort_output,
                atol=absolute_tolerance,
                rtol=relative_tolerance,
            )
        )
        comparison = {
            "name": output_name,
            "shape": list(ort_output.shape),
            "max_abs_error": maximum_error,
            "absolute_tolerance": absolute_tolerance,
            "relative_tolerance": relative_tolerance,
            "parity": parity,
        }
        comparisons.append(comparison)
        if not parity:
            raise RuntimeError(f"Parity failed for {output_name}: {comparison}")
    return comparisons


def add_metadata(model_path: Path, metadata: dict[str, Any]) -> None:
    graph_model = onnx.load(model_path)
    del graph_model.metadata_props[:]
    for key, value in metadata.items():
        property_entry = graph_model.metadata_props.add()
        property_entry.key = key
        property_entry.value = json.dumps(value, sort_keys=True) if not isinstance(value, str) else value
    onnx.save(graph_model, model_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "model" / "artifacts" / "phase3" / "multi_exit.onnx",
    )
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--verification-samples", type=int, default=8)
    return parser.parse_args()


def select_class_diverse_batch(validation_dataset, sample_count: int):
    selected_indices: list[int] = []
    seen_targets: set[int] = set()
    for index, (_, target) in enumerate(validation_dataset.samples):
        if target not in seen_targets:
            selected_indices.append(index)
            seen_targets.add(target)
            if len(selected_indices) == sample_count:
                break
    if len(selected_indices) < sample_count:
        selected_set = set(selected_indices)
        selected_indices.extend(
            index
            for index in range(len(validation_dataset))
            if index not in selected_set
        )
        selected_indices = selected_indices[:sample_count]
    samples = [validation_dataset[index] for index in selected_indices]
    inputs = torch.stack([sample[0] for sample in samples])
    targets = torch.tensor([sample[1] for sample in samples], dtype=torch.int64)
    paths = [validation_dataset.samples[index][0] for index in selected_indices]
    return inputs, targets, paths


def main() -> None:
    args = parse_args()
    if args.verification_samples <= 0:
        raise ValueError("verification-samples must be positive")
    checkpoint_path = args.checkpoint.resolve()
    output_path = args.output.resolve()
    report_path = (
        args.report.resolve()
        if args.report is not None
        else output_path.with_suffix(".verification.json")
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["run_config"]
    if checkpoint["epoch"] != 5:
        raise RuntimeError(f"Expected accepted best checkpoint at epoch 5, found {checkpoint['epoch']}")
    model = MultiExitResNet(
        num_classes=len(config["class_to_idx"]),
        channels=tuple(config["channels"]),
        blocks_per_stage=tuple(config["blocks_per_stage"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    dataset_root, dataset_sha256 = download_imagenette(args.data_root.resolve())
    _, validation_dataset = create_datasets(dataset_root, int(config["arguments"]["image_size"]))
    if validation_dataset.class_to_idx != config["class_to_idx"]:
        raise RuntimeError("Checkpoint and Imagenette class mappings differ")
    verification_inputs, verification_targets, verification_paths = select_class_diverse_batch(
        validation_dataset, args.verification_samples
    )

    export_four_output_model(model, verification_inputs[:1], output_path)
    checkpoint_sha256 = sha256_file(checkpoint_path)
    add_metadata(
        output_path,
        {
            "aeronode_phase": "3",
            "source_checkpoint": checkpoint_path.name,
            "source_checkpoint_sha256": checkpoint_sha256,
            "checkpoint_epoch": str(checkpoint["epoch"]),
            "dataset_archive_sha256": dataset_sha256,
            "class_to_idx": config["class_to_idx"],
            "exit_weights": config["exit_weights"],
            "image_size": str(config["arguments"]["image_size"]),
        },
    )
    graph = validate_graph(output_path, class_count=len(validation_dataset.classes))
    parity = compare_outputs(model, output_path, verification_inputs)
    report = {
        "model_path": str(output_path),
        "model_size_bytes": output_path.stat().st_size,
        "model_sha256": sha256_file(output_path),
        "source_checkpoint": str(checkpoint_path),
        "source_checkpoint_sha256": checkpoint_sha256,
        "source_checkpoint_epoch": checkpoint["epoch"],
        "verification_sample_count": len(verification_targets),
        "verification_targets": verification_targets.tolist(),
        "verification_paths": verification_paths,
        "graph": graph,
        "parity": parity,
        "netron": "not installed; reproducible ONNX checker/shape/runtime inspection completed",
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
