# Phase 2 Validation Record

## Outcome

The compact ResNet-style multi-exit model was trained on the complete official Imagenette 160-pixel training split and independently evaluated at every head on the complete validation split. All four heads learned useful classifiers, and the observed accuracy increased with inference depth.

## Reproducibility

- Dataset: official `imagenette2-160.tgz`
- Archive SHA-256: `64d0c4859f35a461889e0147755a999a48b49bf38a7e0f9bd27003f10db02fe5`
- Samples: 9,469 train / 3,925 validation
- Initialization: random (no pretrained weights)
- Seed: 20260916
- Input: 128x128 crops from Imagenette 160-pixel images
- Backbone: four residual stages, two basic blocks per stage, channels 32/64/128/256
- Exit weights: 0.15 / 0.20 / 0.25 / 0.40
- Optimizer: SGD, learning rate 0.05, momentum 0.9, weight decay 0.0005
- Schedule: cosine annealing over 5 epochs
- Batch size: 64
- Runtime: CPU-only PyTorch 2.14.0 / torchvision 0.29.0

## Independent Validation Results

The best checkpoint (epoch 5) was loaded into a new model instance and evaluated in a validation-only process.

| Exit | Cross-entropy loss | Correct | Samples | Top-1 accuracy |
|---:|---:|---:|---:|---:|
| 1 | 1.378888 | 2,248 | 3,925 | 57.27% |
| 2 | 1.053556 | 2,619 | 3,925 | 66.73% |
| 3 | 0.875750 | 2,818 | 3,925 | 71.80% |
| 4 (final) | 0.829955 | 2,869 | 3,925 | 73.10% |

Weighted validation loss was 0.968464. The fresh validation pass reproduced the metrics stored by training exactly.

## Interpretation and Limitations

These results validate the Phase-2 architecture and every intermediate classifier, not an accuracy optimum. Five CPU epochs were chosen as a bounded prototype experiment. Exit 1 is materially less accurate than later exits, while exits 2 and 3 provide progressively stronger intermediate predictions. Entropy threshold calibration and accuracy/latency trade-off evaluation belong to later phases and have not been performed here.

No multi-exit ONNX export or C++ early-exit integration was performed during Phase 2.
