"""Generate ignored Phase-5D image/reference and adversarial ONNX fixtures.

Uses the same deterministic class-diverse validation selection as Phase 5B.
Python is test orchestration only; every staged request runs in the C++ engine.
"""
from pathlib import Path
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model.export.export_multi_exit import select_class_diverse_batch
from model.training.imagenette_data import create_datasets

OUT = ROOT / "tmp" / "phase5d"


def save_model(model, name):
    onnx.checker.check_model(model)
    onnx.save(model, OUT / name)


def boundary_fixture(name, output_name, mode):
    model = onnx.load(ROOT / "model/artifacts/phase5/stage_2.onnx")
    # Keep the real stage graph, adding a data-dependent boundary transform.
    # ORT cannot infer the runtime batch length statically.
    for node in model.graph.node:
        for i, value in enumerate(node.input):
            if value == output_name:
                node.input[i] = "original_boundary"
        for i, output in enumerate(node.output):
            if output == output_name:
                node.output[i] = "original_boundary"
    constants = {"axis0": np.array(0, dtype=np.int64)}
    if mode in ("failure", "shape"):
        constants.update(one=np.array(1, dtype=np.int64), zero=np.array(0, dtype=np.int64))
    model.graph.initializer.extend(numpy_helper.from_array(v, k) for k, v in constants.items())
    model.graph.node.extend([
        helper.make_node("Shape", ["feat_1"], ["input_shape"]),
        helper.make_node("Gather", ["input_shape", "axis0"], ["batch"], axis=0),
    ])
    if mode == "failure":
        model.graph.node.extend([
            helper.make_node("Range", ["zero", "batch", "one"], ["indices"]),
            helper.make_node("Add", ["indices", "batch"], ["bad_indices"]),
            helper.make_node("Gather", ["original_boundary", "bad_indices"], [output_name], axis=0),
        ])
    elif mode == "shape":
        model.graph.node.extend([
            helper.make_node("Add", ["batch", "one"], ["extra_batch"]),
            helper.make_node("Range", ["zero", "extra_batch", "one"], ["indices"]),
            helper.make_node("Mod", ["indices", "batch"], ["repeat_indices"]),
            helper.make_node("Gather", ["original_boundary", "repeat_indices"], [output_name], axis=0),
        ])
    else:
        model.graph.initializer.append(numpy_helper.from_array(np.array(0, dtype=np.float32), "float_zero"))
        model.graph.node.append(helper.make_node("Div", ["original_boundary", "float_zero"], [output_name]))
    save_model(model, name)


def contract_fixture(name, feature_suffix=(64, 16, 16), logits_suffix=(10,),
                     output_type=TensorProto.FLOAT, batch="batch_size", input_suffix=(32, 32, 32)):
    nodes = [helper.make_node("Shape", ["feat_1"], ["input_shape"]),
             helper.make_node("Gather", ["input_shape", "batch_axis"], ["batch_shape"], axis=0)]
    initializers = [numpy_helper.from_array(np.array([0], np.int64), "batch_axis")]
    outputs = []
    for i, (output, suffix) in enumerate((("feat_2", feature_suffix), ("exit_2_logits", logits_suffix))):
        initializers.append(numpy_helper.from_array(np.array(suffix, np.int64), f"suffix_{i}"))
        nodes.extend([
            helper.make_node("Concat", ["batch_shape", f"suffix_{i}"], [f"shape_{i}"], axis=0),
            helper.make_node("ConstantOfShape", [f"shape_{i}"], [output],
                             value=helper.make_tensor("value", output_type, [1], [0])),
        ])
        outputs.append(helper.make_tensor_value_info(output, output_type, [batch, *suffix]))
    graph = helper.make_graph(nodes, name,
                              [helper.make_tensor_value_info("feat_1", TensorProto.FLOAT, [batch, *input_suffix])],
                              outputs, initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8)
    original = onnx.load(ROOT / "model/artifacts/phase5/stage_2.onnx")
    model.metadata_props.extend(original.metadata_props)
    save_model(model, name)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    _, validation = create_datasets(ROOT / "data/imagenette2-160", 128)
    inputs, _, _ = select_class_diverse_batch(validation, 10)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(ROOT / "model/artifacts/phase3/multi_exit_imagenette.onnx"),
                                   sess_options=options, providers=["CPUExecutionProvider"])
    for batch in (1, 3, 5):
        data = inputs[:batch].numpy().astype(np.float32)
        data.tofile(OUT / f"input_{batch}.bin")
        for i, output in enumerate(session.run(None, {"input": data}), 1):
            output.tofile(OUT / f"reference_{batch}_{i}.bin")
    for output, prefix in (("feat_2", "feature"), ("exit_2_logits", "logits")):
        boundary_fixture(f"bad_{prefix}_shape.onnx", output, "shape")
        boundary_fixture(f"bad_{prefix}_finite.onnx", output, "finite")
    boundary_fixture("run_failure.onnx", "feat_2", "failure")
    contract_fixture("bad_feature_metadata.onnx", feature_suffix=(63, 16, 16))
    contract_fixture("bad_logits_metadata.onnx", logits_suffix=(11,))
    contract_fixture("bad_rank_metadata.onnx", logits_suffix=(1, 10))
    contract_fixture("bad_type_metadata.onnx", output_type=TensorProto.DOUBLE)
    contract_fixture("bad_batch_metadata.onnx", batch=1)
    contract_fixture("bad_input_metadata.onnx", input_suffix=(31, 32, 32))
    model = onnx.load(ROOT / "model/artifacts/phase5/stage_2.onnx")
    del model.graph.output[1]
    save_model(model, "bad_output_count.onnx")
    for key, value, name in (
        ("stage_index", "3", "bad_stage_identity.onnx"),
        ("dynamic_batch", "false", "bad_dynamic_property.onnx"),
        ("source_checkpoint_sha256", "invalid", "bad_hash_metadata.onnx"),
        ("source_checkpoint_sha256", "0" * 64, "mixed_checkpoint.onnx"),
    ):
        modified = onnx.load(ROOT / "model/artifacts/phase5/stage_2.onnx")
        for entry in modified.metadata_props:
            if entry.key == key:
                entry.value = value
        save_model(modified, name)
    modified = onnx.load(ROOT / "model/artifacts/phase5/stage_2.onnx")
    del modified.metadata_props[:]
    save_model(modified, "missing_metadata.onnx")
    (OUT / "corrupt.onnx").write_bytes(b"not an ONNX protobuf")
    print(f"Phase 5D fixtures ready: {OUT}")


if __name__ == "__main__":
    main()
