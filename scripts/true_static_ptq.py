#!/usr/bin/env python3
"""
TRUE Post-Training Quantization (PTQ) using ONNX Runtime's static quantization.
This performs real INT8 quantization with calibration data, producing QuantizeLinear
and DequantizeLinear ops in the graph — resulting in actual INT8 speedups.
"""
import os, sys, time, cv2
import numpy as np
from pathlib import Path

import onnxruntime as ort
from onnxruntime.quantization import (
    quantize_static,
    CalibrationDataReader,
    QuantType,
    QuantFormat,
)

# ---- Config ----
FP32_ONNX   = '/home/ls5255/FastSAM/runs/segment/pupil_retrain/weights/best.onnx'
INT8_ONNX   = '/home/ls5255/FastSAM/runs/segment/pupil_retrain/weights/best_int8_static.onnx'
CALIB_DIR   = '/home/ls5255/FastSAM/dataset_pupil_seg/images/train'
TEST_IMAGE  = '/home/ls5255/Documents/outputs_batch_cam3_video_range_3001_3771/overlay__03005.png'
IMGSZ       = 1024
NUM_CALIB   = 50   # number of calibration images
WARMUP      = 10
RUNS        = 30


class FastSAMCalibrationReader(CalibrationDataReader):
    """Feeds preprocessed calibration images to the quantizer."""
    def __init__(self, calib_dir, input_name, imgsz=1024, max_images=50):
        self.input_name = input_name
        self.imgsz = imgsz
        exts = {'.png', '.jpg', '.jpeg'}
        self.image_paths = [
            os.path.join(calib_dir, f) 
            for f in sorted(os.listdir(calib_dir)) 
            if Path(f).suffix.lower() in exts
        ][:max_images]
        self.index = 0
        print(f"Calibration reader: {len(self.image_paths)} images from {calib_dir}")

    def get_next(self):
        if self.index >= len(self.image_paths):
            return None
        img_path = self.image_paths[self.index]
        self.index += 1
        
        # Preprocess: resize, normalize, CHW, add batch dim
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.imgsz, self.imgsz))
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
        img = np.expand_dims(img, axis=0)    # Add batch dim
        
        if self.index % 10 == 0:
            print(f"  Calibrating... {self.index}/{len(self.image_paths)}")
        
        return {self.input_name: img}

    def rewind(self):
        self.index = 0


def preprocess_image(path, imgsz=1024):
    """Preprocess a single image for inference."""
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (imgsz, imgsz))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img


def benchmark_ort(session, input_name, test_data, name, warmup=10, runs=30):
    """Benchmark an ORT session."""
    for _ in range(warmup):
        session.run(None, {input_name: test_data})
    
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        session.run(None, {input_name: test_data})
        times.append((time.perf_counter() - t0) * 1000)
    
    mean_t = np.mean(times)
    std_t = np.std(times)
    fps = 1000.0 / mean_t
    print(f"  {name:40s}  {mean_t:8.2f} ms ± {std_t:5.2f} ms   ({fps:.2f} FPS)")
    return mean_t, std_t, fps


# ==== Step 1: Get input name from FP32 model ====
print("="*70)
print("REAL STATIC INT8 QUANTIZATION")
print("="*70)

sess_options = ort.SessionOptions()
fp32_session = ort.InferenceSession(FP32_ONNX, sess_options, providers=['CPUExecutionProvider'])
input_name = fp32_session.get_inputs()[0].name
input_shape = fp32_session.get_inputs()[0].shape
print(f"Input name: {input_name}, shape: {input_shape}")

# ==== Step 2: Run static quantization with calibration ====
print(f"\nQuantizing {FP32_ONNX} -> {INT8_ONNX}")
print(f"Using {NUM_CALIB} calibration images from {CALIB_DIR}")

calib_reader = FastSAMCalibrationReader(CALIB_DIR, input_name, IMGSZ, NUM_CALIB)

quantize_static(
    model_input=FP32_ONNX,
    model_output=INT8_ONNX,
    calibration_data_reader=calib_reader,
    quant_format=QuantFormat.QDQ,          # QDQ format (QuantizeLinear/DequantizeLinear)
    per_channel=True,                       # Per-channel quantization for better accuracy
    weight_type=QuantType.QInt8,            # INT8 weights
    activation_type=QuantType.QUInt8,       # UINT8 activations
)

print(f"\nQuantized model saved to: {INT8_ONNX}")
print(f"FP32 size: {os.path.getsize(FP32_ONNX)/1e6:.1f} MB")
print(f"INT8 size: {os.path.getsize(INT8_ONNX)/1e6:.1f} MB")
print(f"Compression: {os.path.getsize(FP32_ONNX)/os.path.getsize(INT8_ONNX):.2f}x")

# ==== Step 3: Verify quantization ops exist ====
import onnx
from collections import Counter
int8_model = onnx.load(INT8_ONNX)
op_types = Counter(node.op_type for node in int8_model.graph.node)
quant_ops = {k: v for k, v in op_types.items() if 'quantize' in k.lower() or 'qlinear' in k.lower()}
print(f"\nQuantization ops in INT8 model: {dict(quant_ops)}")
dtypes = Counter(init.data_type for init in int8_model.graph.initializer)
print(f"Weight data types: {dict(dtypes)}")
print(f"  (3=INT8, 2=UINT8, 1=FP32)")

# ==== Step 4: Benchmark FP32 vs TRUE INT8 ====
print("\n" + "="*70)
print("BENCHMARKING FP32 vs TRUE INT8 (ONNX Runtime, CPU)")
print("="*70)

test_data = preprocess_image(TEST_IMAGE, IMGSZ)

# FP32 on CPU
fp32_cpu = ort.InferenceSession(FP32_ONNX, providers=['CPUExecutionProvider'])
fp32_mean, fp32_std, fp32_fps = benchmark_ort(fp32_cpu, input_name, test_data, "FP32 ONNX (CPU)")

# INT8 on CPU
int8_cpu = ort.InferenceSession(INT8_ONNX, providers=['CPUExecutionProvider'])
int8_cpu_mean, int8_cpu_std, int8_cpu_fps = benchmark_ort(int8_cpu, input_name, test_data, "TRUE INT8 Static PTQ (CPU)")

print(f"\n  CPU Speedup: {fp32_mean/int8_cpu_mean:.2f}x")

# FP32 on GPU
print("\n" + "-"*70)
print("BENCHMARKING FP32 vs TRUE INT8 (ONNX Runtime, GPU)")
print("-"*70)

fp32_gpu = ort.InferenceSession(FP32_ONNX, providers=['CUDAExecutionProvider'])
fp32_gpu_mean, fp32_gpu_std, fp32_gpu_fps = benchmark_ort(fp32_gpu, input_name, test_data, "FP32 ONNX (GPU)")

int8_gpu = ort.InferenceSession(INT8_ONNX, providers=['CUDAExecutionProvider'])
int8_gpu_mean, int8_gpu_std, int8_gpu_fps = benchmark_ort(int8_gpu, input_name, test_data, "TRUE INT8 Static PTQ (GPU)")

print(f"\n  GPU Speedup: {fp32_gpu_mean/int8_gpu_mean:.2f}x")

# ==== Summary ====
print("\n" + "="*70)
print("FINAL SUMMARY")
print("="*70)
results = {
    'FP32 ONNX (CPU)': (fp32_mean, fp32_std, fp32_fps),
    'INT8 Static PTQ (CPU)': (int8_cpu_mean, int8_cpu_std, int8_cpu_fps),
    'FP32 ONNX (GPU)': (fp32_gpu_mean, fp32_gpu_std, fp32_gpu_fps),
    'INT8 Static PTQ (GPU)': (int8_gpu_mean, int8_gpu_std, int8_gpu_fps),
}

out_file = '/home/ls5255/FastSAM/paper_figures_inference/real_ptq_proof/true_int8_benchmark.txt'
with open(out_file, 'w') as f:
    f.write("TRUE STATIC INT8 PTQ BENCHMARK\n")
    f.write(f"FP32 model: {FP32_ONNX}\n")
    f.write(f"INT8 model: {INT8_ONNX}\n")
    f.write(f"FP32 size: {os.path.getsize(FP32_ONNX)/1e6:.1f} MB\n")
    f.write(f"INT8 size: {os.path.getsize(INT8_ONNX)/1e6:.1f} MB\n\n")
    for name, (m, s, fps) in results.items():
        line = f"{name:35s}  {m:8.2f} ± {s:5.2f} ms  ({fps:.2f} FPS)"
        print(f"  {line}")
        f.write(line + "\n")
    f.write(f"\nCPU Speedup: {fp32_mean/int8_cpu_mean:.2f}x\n")
    f.write(f"GPU Speedup: {fp32_gpu_mean/int8_gpu_mean:.2f}x\n")

print(f"\nSaved: {out_file}")
