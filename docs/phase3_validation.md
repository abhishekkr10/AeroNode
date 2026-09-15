# Phase 3 ONNX Validation Record

## Artifact and Provenance

- Model: `model/artifacts/phase3/multi_exit_imagenette.onnx`
- Size: 11,217,740 bytes (10.70 MiB)
- SHA-256: `0cc08e571383279798c00019f0079bcbc107db49a1fe8efca444a4b754f35051`
- Source: accepted Phase-2 `best_final_exit.pt`, epoch 5
- Source checkpoint SHA-256: `32fb2975bcfe14d7311bc91db1553442391468d52c74f34a4efeb9d2aef27366`
- ONNX opset: 17
- ONNX IR version: 8

The export embeds checkpoint, dataset, class-map, exit-weight, and image-size provenance in ONNX metadata. The validated Phase-2 checkpoint was retained unchanged.

## Graph Structure

ONNX checker full validation and strict shape inference passed.

| Kind | Name | Shape |
|---|---|---|
| Input | `input` | `[batch_size, 3, 128, 128]` |
| Output | `exit_1_logits` | `[batch_size, 10]` |
| Output | `exit_2_logits` | `[batch_size, 10]` |
| Output | `exit_3_logits` | `[batch_size, 10]` |
| Output | `exit_4_logits` | `[batch_size, 10]` |

The graph contains 58 nodes and 48 initializers. Operator inventory: 20 Conv, 17 Relu, 8 Add, 4 GlobalAveragePool, 4 Flatten, 4 Gemm, and 1 MaxPool. The four GlobalAveragePool/Flatten/Gemm paths confirm that all four classifier heads are retained.

## Numerical Parity

PyTorch and ONNX Runtime CPU execution used the same deterministic batch of ten validation images, one from each Imagenette class.

Tolerance: absolute `1e-5`, relative `1e-4`.

| Output | Runtime shape | Maximum absolute error | Result |
|---|---:|---:|---|
| `exit_1_logits` | `[10, 10]` | `2.3841858e-06` | Pass |
| `exit_2_logits` | `[10, 10]` | `9.5367432e-07` | Pass |
| `exit_3_logits` | `[10, 10]` | `1.4305115e-06` | Pass |
| `exit_4_logits` | `[10, 10]` | `1.6689301e-06` | Pass |

## Representation Decision

This monolithic four-output graph is the fidelity/reference artifact for the next C++ ONNX Runtime phase. It preserves all trained exits and supports dynamic batch sizes, but it is not claimed to implement compute-saving early termination. Later early-exit execution will require separate stage graphs or another explicitly supported execution boundary. Those later graphs must preserve parity with this reference and PyTorch.

Netron was not installed. Programmatic inspection used ONNX checker, strict shape inference, metadata inspection, operator inventory, and ONNX Runtime execution. The artifact remains directly openable in Netron if visual inspection is desired later.
