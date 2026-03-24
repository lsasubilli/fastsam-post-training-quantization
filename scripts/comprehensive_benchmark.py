#!/usr/bin/env python3
"""
Thorough FP32 vs FP16 vs ONNX comparison benchmark.
Tests PyTorch native and ONNX Runtime with proper CUDA synchronization.
"""
import time, os, sys
import numpy as np
import torch

sys.path.insert(0, '/home/ls5255/FastSAM')
from ultralytics import YOLO

WEIGHTS   = '/home/ls5255/FastSAM/runs/segment/pupil_retrain/weights/best.pt'
ONNX_PATH = '/home/ls5255/FastSAM/runs/segment/pupil_retrain/weights/best.onnx'
TEST_IMG  = '/home/ls5255/Documents/outputs_batch_cam3_video_range_3001_3771/overlay__03005.png'
WARMUP    = 10
RUNS      = 30

def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def bench(model, name, device='0'):
    # Warmup
    for _ in range(WARMUP):
        model.predict(source=TEST_IMG, imgsz=1024, verbose=False, device=device)
        sync()

    times = []
    for _ in range(RUNS):
        sync()
        t0 = time.perf_counter()
        model.predict(source=TEST_IMG, imgsz=1024, verbose=False, device=device)
        sync()
        times.append((time.perf_counter() - t0) * 1000.0)

    mean_t = float(np.mean(times))
    std_t  = float(np.std(times))
    fps    = 1000.0 / mean_t
    print(f"  {name:40s}  {mean_t:8.2f} ms ± {std_t:5.2f} ms   ({fps:.2f} FPS)")
    return mean_t, std_t, fps

print("="*80)
print("COMPREHENSIVE FastSAM INFERENCE BENCHMARK")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
print(f"Warmup: {WARMUP}, Runs: {RUNS}")
print(f"Input: {TEST_IMG}")
print("="*80)

results = {}

# --- 1. PyTorch FP32 on GPU ---
print("\n[1/4] PyTorch FP32 (GPU)...")
m = YOLO(WEIGHTS)
results['PyTorch FP32 (GPU)'] = bench(m, 'PyTorch FP32 (GPU)', device='0')
del m
torch.cuda.empty_cache()

# --- 2. PyTorch FP16 (half) on GPU ---
print("\n[2/4] PyTorch FP16 half (GPU)...")
m = YOLO(WEIGHTS)
results['PyTorch FP16 half (GPU)'] = bench(m, 'PyTorch FP16 half (GPU)', device='0')
del m
torch.cuda.empty_cache()

# --- 3. ONNX FP32 on GPU (via ORT CUDA provider) ---
# First export a clean FP32 ONNX if the existing one was INT8
print("\n[3/4] ONNX (GPU via ORT CUDAExecutionProvider)...")
m = YOLO(ONNX_PATH, task='segment')
results['ONNX (GPU ORT)'] = bench(m, 'ONNX (GPU ORT)', device='0')
del m
torch.cuda.empty_cache()

# --- 4. ONNX on CPU ---
print("\n[4/4] ONNX (CPU via ORT CPUExecutionProvider)...")
m = YOLO(ONNX_PATH, task='segment')
results['ONNX (CPU ORT)'] = bench(m, 'ONNX (CPU ORT)', device='cpu')
del m

# --- Summary Table ---
print("\n" + "="*80)
print(f"{'Configuration':40s}  {'Latency':>12s}  {'FPS':>10s}")
print("-"*80)
for name, (mean, std, fps) in results.items():
    print(f"  {name:40s}  {mean:7.2f}±{std:4.1f} ms  {fps:8.2f}")
print("="*80)

# Save to file
out = '/home/ls5255/FastSAM/paper_figures_inference/real_ptq_proof/comprehensive_benchmark.txt'
with open(out, 'w') as f:
    f.write("COMPREHENSIVE FastSAM INFERENCE BENCHMARK\n")
    f.write(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}\n")
    f.write(f"Warmup: {WARMUP}, Runs: {RUNS}\n\n")
    f.write(f"{'Configuration':40s}  {'Latency':>12s}  {'FPS':>10s}\n")
    f.write("-"*70 + "\n")
    for name, (mean, std, fps) in results.items():
        f.write(f"  {name:40s}  {mean:7.2f}±{std:4.1f} ms  {fps:8.2f}\n")
print(f"\nSaved: {out}")
