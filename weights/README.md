# Model Weights

The model weights are too large for GitHub's file size limits.

## Required Files

| File | Size | Description |
|---|---|---|
| `best.pt` | 823 MB | Fine-tuned FP32 PyTorch checkpoint (YOLOv8x-seg) |
| `best.onnx` | 274 MB | INT8 Post-Training Quantized ONNX model |

## How to Obtain

### From the Training Machine

```bash
scp user@training-machine:/home/ls5255/FastSAM/runs/segment/pupil_retrain/weights/best.pt weights/
scp user@training-machine:/home/ls5255/FastSAM/runs/segment/pupil_retrain/weights/best.onnx weights/
```

### Re-Export INT8 ONNX from FP32

If you only have `best.pt`, you can regenerate the INT8 ONNX model:

```bash
python scripts/real_ptq_benchmark.py
```

This will:
1. Load `best.pt`
2. Run calibration over the training dataset
3. Export `best.onnx` with INT8 quantization
