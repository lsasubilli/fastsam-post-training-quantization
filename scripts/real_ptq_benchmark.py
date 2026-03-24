#!/usr/bin/env python3
import time
import os
import cv2
import numpy as np
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    print("Install ultralytics: pip install ultralytics")
    exit(1)

model_path = '/home/ls5255/FastSAM/runs/segment/pupil_retrain/weights/best.pt'
onnx_int8_path = '/home/ls5255/FastSAM/runs/segment/pupil_retrain/weights/best.onnx' # From prev run
test_image = '/home/ls5255/Documents/outputs_batch_cam3_video_range_3001_3771/overlay__03005.png'

print("Loading FastSAM (YOLOv8x-seg) FP32...")
model = YOLO(model_path)

def benchmark_model(model_obj, name, runs=20, warmup=5):
    # Warmup
    for _ in range(warmup):
        model_obj.predict(source=test_image, imgsz=1024, verbose=False, device='0')
    
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        model_obj.predict(source=test_image, imgsz=1024, verbose=False, device='0')
        times.append((time.perf_counter() - t0) * 1000)
        
    mean_t = np.mean(times)
    std_t = np.std(times)
    print(f"{name} latency: {mean_t:.2f} ms ± {std_t:.2f} ms")
    return mean_t, std_t

print("\n--- Benchmarking Latency ---")
fp32_mean, fp32_std = benchmark_model(model, "PyTorch FP32")

# Bench ONNX INT8
try:
    print(f"\nLoading ONNX INT8 model from {onnx_int8_path}...")
    model_int8 = YOLO(onnx_int8_path, task='segment')
    int8_mean, int8_std = benchmark_model(model_int8, "ONNX INT8 PTQ")
    print(f"Speedup: {fp32_mean / int8_mean:.2f}x")
except Exception as e:
    print(f"Failed to bench INT8: {e}")

print("\nGenerating visual proofs...")
out_dir = '/home/ls5255/FastSAM/paper_figures_inference/real_ptq_proof'
os.makedirs(out_dir, exist_ok=True)
out_fp32 = os.path.join(out_dir, 'fp32_cpu.jpg')
out_int8 = os.path.join(out_dir, 'int8_gpu.jpg')

res_fp32 = model.predict(source=test_image, imgsz=1024, verbose=False, device='0')
cv2.imwrite(out_fp32, res_fp32[0].plot())
print(f"Saved FP32 image to {out_fp32}")

try:
    res_int8 = model_int8.predict(source=test_image, imgsz=1024, verbose=False, device='0')
    cv2.imwrite(out_int8, res_int8[0].plot())
    print(f"Saved INT8 image to {out_int8}")
except Exception as e:
    print(f"Failed to save INT8 image: {e}")

# Save Summary
with open(os.path.join(out_dir, 'real_ptq_latency_report.txt'), 'w') as f:
    f.write(f"FP32 PyTorch Model Latency: {fp32_mean:.2f} ms ± {fp32_std:.2f} ms\n")
    if hasattr(globals(), 'int8_mean') or 'int8_mean' in locals():
        f.write(f"INT8 ONNX Model Latency: {int8_mean:.2f} ms ± {int8_std:.2f} ms\n")
        f.write(f"Speedup vs FP32 Baseline: {fp32_mean / int8_mean:.2f}x\n")
print(f"Wrote benchmark report to {out_dir}/real_ptq_latency_report.txt")
