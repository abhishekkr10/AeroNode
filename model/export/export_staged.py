"""Export the validated multi-exit checkpoint as independently executable ONNX stages.

Phase 5A milestone: exports stage_1.onnx through stage_4.onnx, performs ONNX
structural and shape validation, verifies session loadability, calculates analytical
FLOP/MAC estimates, and generates staged_export_manifest.json.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import onnx
from onnx import shape_inference
import onnxruntime as ort
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.training.multi_exit_resnet import MultiExitResNet

OPSET_VERSION = 17


class Stage1Module(nn.Module):
    """Stem + Backbone Stage 0 + Exit Head 0."""

    def __init__(self, stem: nn.Module, stage: nn.Module, head: nn.Module) -> None:
        super().__init__()
        self.stem = stem
        self.stage = stage
        self.head = head

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.stage(self.stem(inputs))
        logits = self.head(features)
        return features, logits


class Stage2Module(nn.Module):
    """Backbone Stage 1 + Exit Head 1."""

    def __init__(self, stage: nn.Module, head: nn.Module) -> None:
        super().__init__()
        self.stage = stage
        self.head = head

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.stage(features)
        logits = self.head(features)
        return features, logits


class Stage3Module(nn.Module):
    """Backbone Stage 2 + Exit Head 2."""

    def __init__(self, stage: nn.Module, head: nn.Module) -> None:
        super().__init__()
        self.stage = stage
        self.head = head

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.stage(features)
        logits = self.head(features)
        return features, logits


class Stage4Module(nn.Module):
    """Backbone Stage 3 + Exit Head 3 (Terminal Exit).

    Outputs exit_4_logits only. No intermediate feature tensor is emitted.
    """

    def __init__(self, stage: nn.Module, head: nn.Module) -> None:
        super().__init__()
        self.stage = stage
        self.head = head

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = self.stage(features)
        logits = self.head(features)
        return logits


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def count_stage_flops(module: nn.Module, dummy_input: torch.Tensor) -> dict[str, int]:
    """Calculate theoretical MACs and FLOPs through analytical layer hook counting.

    These are theoretical algorithmic estimates (Conv2d, Linear) and NOT measured
    hardware execution performance or runtime latency.
    """
    macs = 0
    flops = 0

    def conv_hook(
        layer: nn.Conv2d,
        _layer_input: tuple[torch.Tensor, ...],
        layer_output: torch.Tensor,
    ) -> None:
        nonlocal macs, flops
        cin = layer.in_channels
        cout = layer.out_channels
        kh, kw = layer.kernel_size
        _, _, hout, wout = layer_output.shape
        layer_macs = cin * cout * kh * kw * hout * wout
        layer_flops = 2 * layer_macs
        if layer.bias is not None:
            layer_flops += cout * hout * wout
        macs += layer_macs
        flops += layer_flops

    def linear_hook(
        layer: nn.Linear,
        _layer_input: tuple[torch.Tensor, ...],
        _layer_output: torch.Tensor,
    ) -> None:
        nonlocal macs, flops
        layer_macs = layer.in_features * layer.out_features
        layer_flops = 2 * layer_macs
        if layer.bias is not None:
            layer_flops += layer.out_features
        macs += layer_macs
        flops += layer_flops

    hooks = []
    for sub_module in module.modules():
        if isinstance(sub_module, nn.Conv2d):
            hooks.append(sub_module.register_forward_hook(conv_hook))
        elif isinstance(sub_module, nn.Linear):
            hooks.append(sub_module.register_forward_hook(linear_hook))

    with torch.no_grad():
        module(dummy_input)

    for hook in hooks:
        hook.remove()

    return {"analytical_macs": macs, "analytical_flops": flops}


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


def add_stage_metadata(model_path: Path, metadata: dict[str, Any]) -> None:
    graph_model = onnx.load(model_path)
    del graph_model.metadata_props[:]
    for key, value in metadata.items():
        entry = graph_model.metadata_props.add()
        entry.key = key
        entry.value = json.dumps(value, sort_keys=True) if not isinstance(value, str) else value
    onnx.save(graph_model, model_path)


def validate_stage_graph(
    model_path: Path,
    expected_inputs: dict[str, list[int | str]],
    expected_outputs: dict[str, list[int | str]],
) -> dict[str, Any]:
    """Run full ONNX checker and strict shape inference on an exported stage graph."""
    graph_model = onnx.load(model_path)
    onnx.checker.check_model(graph_model, full_check=True)
    inferred_model = shape_inference.infer_shapes(graph_model, strict_mode=True, data_prop=True)

    inputs = {value.name: value_shape(value) for value in inferred_model.graph.input}
    outputs = {value.name: value_shape(value) for value in inferred_model.graph.output}

    if inputs != expected_inputs:
        raise RuntimeError(
            f"Input contract mismatch for {model_path.name}: expected {expected_inputs}, got {inputs}"
        )
    if outputs != expected_outputs:
        raise RuntimeError(
            f"Output contract mismatch for {model_path.name}: expected {expected_outputs}, got {outputs}"
        )

    # Verify data types are float32 (type enum 1)
    for inp in inferred_model.graph.input:
        elem_type = inp.type.tensor_type.elem_type
        if elem_type != onnx.TensorProto.FLOAT:
            raise RuntimeError(f"Input {inp.name} in {model_path.name} is not FLOAT (got {elem_type})")
    for out in inferred_model.graph.output:
        elem_type = out.type.tensor_type.elem_type
        if elem_type != onnx.TensorProto.FLOAT:
            raise RuntimeError(f"Output {out.name} in {model_path.name} is not FLOAT (got {elem_type})")

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


def verify_session_loadable(model_path: Path) -> None:
    """Verify that ONNX Runtime can cleanly parse and load the exported stage graph."""
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    if not session.get_inputs() or not session.get_outputs():
        raise RuntimeError(f"Session failed to discover inputs/outputs for {model_path.name}")


def export_staged_models(
    checkpoint_path: Path,
    output_dir: Path,
    opset_version: int = OPSET_VERSION,
) -> dict[str, Any]:
    """Partition MultiExitResNet into four ONNX stages and validate contracts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_sha256 = sha256_file(checkpoint_path)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["run_config"]
    if checkpoint["epoch"] != 5:
        raise RuntimeError(f"Expected accepted checkpoint at epoch 5, found {checkpoint['epoch']}")

    model = MultiExitResNet(
        num_classes=len(config["class_to_idx"]),
        channels=tuple(config["channels"]),
        blocks_per_stage=tuple(config["blocks_per_stage"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    # Stage specifications
    stage_specs = [
        {
            "index": 1,
            "name": "stage_1",
            "file": "stage_1.onnx",
            "module": Stage1Module(model.stem, model.stages[0], model.exit_heads[0]),
            "dummy_input": torch.randn(1, 3, 128, 128),
            "input_names": ["input"],
            "output_names": ["feat_1", "exit_1_logits"],
            "dynamic_axes": {
                "input": {0: "batch_size"},
                "feat_1": {0: "batch_size"},
                "exit_1_logits": {0: "batch_size"},
            },
            "expected_inputs": {"input": ["batch_size", 3, 128, 128]},
            "expected_outputs": {
                "feat_1": ["batch_size", 32, 32, 32],
                "exit_1_logits": ["batch_size", 10],
            },
            "description": "Stem + Backbone Stage 0 + Exit Head 0",
        },
        {
            "index": 2,
            "name": "stage_2",
            "file": "stage_2.onnx",
            "module": Stage2Module(model.stages[1], model.exit_heads[1]),
            "dummy_input": torch.randn(1, 32, 32, 32),
            "input_names": ["feat_1"],
            "output_names": ["feat_2", "exit_2_logits"],
            "dynamic_axes": {
                "feat_1": {0: "batch_size"},
                "feat_2": {0: "batch_size"},
                "exit_2_logits": {0: "batch_size"},
            },
            "expected_inputs": {"feat_1": ["batch_size", 32, 32, 32]},
            "expected_outputs": {
                "feat_2": ["batch_size", 64, 16, 16],
                "exit_2_logits": ["batch_size", 10],
            },
            "description": "Backbone Stage 1 + Exit Head 1",
        },
        {
            "index": 3,
            "name": "stage_3",
            "file": "stage_3.onnx",
            "module": Stage3Module(model.stages[2], model.exit_heads[2]),
            "dummy_input": torch.randn(1, 64, 16, 16),
            "input_names": ["feat_2"],
            "output_names": ["feat_3", "exit_3_logits"],
            "dynamic_axes": {
                "feat_2": {0: "batch_size"},
                "feat_3": {0: "batch_size"},
                "exit_3_logits": {0: "batch_size"},
            },
            "expected_inputs": {"feat_2": ["batch_size", 64, 16, 16]},
            "expected_outputs": {
                "feat_3": ["batch_size", 128, 8, 8],
                "exit_3_logits": ["batch_size", 10],
            },
            "description": "Backbone Stage 2 + Exit Head 2",
        },
        {
            "index": 4,
            "name": "stage_4",
            "file": "stage_4.onnx",
            "module": Stage4Module(model.stages[3], model.exit_heads[3]),
            "dummy_input": torch.randn(1, 128, 8, 8),
            "input_names": ["feat_3"],
            "output_names": ["exit_4_logits"],
            "dynamic_axes": {
                "feat_3": {0: "batch_size"},
                "exit_4_logits": {0: "batch_size"},
            },
            "expected_inputs": {"feat_3": ["batch_size", 128, 8, 8]},
            "expected_outputs": {
                "exit_4_logits": ["batch_size", 10],
            },
            "description": "Backbone Stage 3 + Exit Head 3 (Terminal Exit)",
        },
    ]

    manifest_stages: list[dict[str, Any]] = []
    total_analytical_macs = 0
    total_analytical_flops = 0

    for spec in stage_specs:
        stage_module = spec["module"].eval()
        dummy_input = spec["dummy_input"]
        stage_path = output_dir / spec["file"]

        # Calculate analytical FLOPs for this stage
        stage_flops = count_stage_flops(stage_module, dummy_input)
        total_analytical_macs += stage_flops["analytical_macs"]
        total_analytical_flops += stage_flops["analytical_flops"]

        # Export to ONNX
        torch.onnx.export(
            stage_module,
            (dummy_input,),
            stage_path,
            input_names=spec["input_names"],
            output_names=spec["output_names"],
            dynamic_axes=spec["dynamic_axes"],
            opset_version=opset_version,
            do_constant_folding=True,
            dynamo=False,
        )

        # Embed metadata
        stage_meta = {
            "aeronode_phase": "5A",
            "stage_index": str(spec["index"]),
            "stage_name": spec["name"],
            "description": spec["description"],
            "source_checkpoint": checkpoint_path.name,
            "source_checkpoint_sha256": checkpoint_sha256,
            "checkpoint_epoch": str(checkpoint["epoch"]),
            "analytical_macs": str(stage_flops["analytical_macs"]),
            "analytical_flops": str(stage_flops["analytical_flops"]),
            "flops_estimation_note": (
                "Theoretical analytical layer estimate (Conv2d, Linear). "
                "Not measured runtime latency or hardware performance."
            ),
            "input_names": spec["input_names"],
            "output_names": spec["output_names"],
            "opset_version": str(opset_version),
            "dynamic_batch": "true",
        }
        add_stage_metadata(stage_path, stage_meta)

        # Validate graph structure and strict shapes
        graph_info = validate_stage_graph(
            stage_path,
            expected_inputs=spec["expected_inputs"],
            expected_outputs=spec["expected_outputs"],
        )

        # Verify loadable by ONNX Runtime session
        verify_session_loadable(stage_path)

        manifest_stages.append(
            {
                "stage_index": spec["index"],
                "stage_name": spec["name"],
                "file_name": spec["file"],
                "relative_path": str(stage_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "file_size_bytes": stage_path.stat().st_size,
                "sha256": sha256_file(stage_path),
                "description": spec["description"],
                "inputs": graph_info["inputs"],
                "outputs": graph_info["outputs"],
                "node_count": graph_info["node_count"],
                "initializer_count": graph_info["initializer_count"],
                "operator_counts": graph_info["operator_counts"],
                "theoretical_macs": stage_flops["analytical_macs"],
                "theoretical_flops": stage_flops["analytical_flops"],
            }
        )

    manifest = {
        "schema_version": "1.0.0",
        "aeronode_phase": "5A",
        "export_timestamp_utc": None,
        "source_checkpoint": {
            "file_name": checkpoint_path.name,
            "relative_path": str(checkpoint_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "sha256": checkpoint_sha256,
            "epoch": checkpoint["epoch"],
            "class_to_idx": config["class_to_idx"],
            "channels": config["channels"],
            "blocks_per_stage": config["blocks_per_stage"],
        },
        "opset_version": opset_version,
        "compute_summary": {
            "total_analytical_macs": total_analytical_macs,
            "total_analytical_flops": total_analytical_flops,
            "total_analytical_mflops": round(total_analytical_flops / 1e6, 2),
            "per_stage_flops_breakdown": {
                f"stage_{s['stage_index']}": {
                    "macs": s["theoretical_macs"],
                    "flops": s["theoretical_flops"],
                    "mflops": round(s["theoretical_flops"] / 1e6, 2),
                    "fraction_of_total": round(s["theoretical_flops"] / total_analytical_flops, 4),
                }
                for s in manifest_stages
            },
            "disclaimer": (
                "Theoretical FLOP and MAC numbers are calculated analytically from layer dimensions "
                "(Conv2d, Linear). They represent algorithmic workload estimates and must not be "
                "equated with measured hardware latency, CPU cycles, or cache execution time."
            ),
        },
        "stages": manifest_stages,
    }

    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export MultiExitResNet into 4 staged ONNX models.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "model" / "artifacts" / "phase2-imagenette-5ep" / "best_final_exit.pt",
        help="Path to trained PyTorch checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "model" / "artifacts" / "phase5",
        help="Output directory for staged ONNX models.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "model" / "artifacts" / "phase5" / "staged_export_manifest.json",
        help="Path for generated export manifest JSON.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=OPSET_VERSION,
        help=f"Target ONNX opset version (default: {OPSET_VERSION}).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = args.checkpoint.resolve()
    output_dir = args.output_dir.resolve()
    manifest_path = args.manifest.resolve()

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"[Phase 5A] Loading checkpoint: {checkpoint_path.name}")
    manifest = export_staged_models(checkpoint_path, output_dir, opset_version=args.opset)

    # Record generation time in ISO 8601
    import datetime
    manifest["export_timestamp_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as manifest_file:
        json.dump(manifest, manifest_file, indent=2)

    print(f"[Phase 5A] Successfully exported 4 staged ONNX models to {output_dir}")
    print(f"[Phase 5A] Manifest written to {manifest_path}")
    print("\n--- Summary of Staged ONNX Artifacts ---")
    for stage in manifest["stages"]:
        print(
            f"  {stage['file_name']:<14} | Size: {stage['file_size_bytes']:>9,} bytes | "
            f"Nodes: {stage['node_count']:>2} | FLOPs: {stage['theoretical_flops']:>11,} "
            f"({stage['theoretical_flops']/1e6:.2f} MFLOPs)"
        )
    print(
        f"\nTotal Analytical FLOPs: {manifest['compute_summary']['total_analytical_flops']:,} "
        f"({manifest['compute_summary']['total_analytical_mflops']} MFLOPs)"
    )


if __name__ == "__main__":
    main()
