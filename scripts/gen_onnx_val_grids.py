#!/usr/bin/env python3
import os, cv2, time
import numpy as np

os.environ['MPLBACKEND'] = 'Agg'
from ultralytics import YOLO

ONNX_PATH = '/home/ls5255/FastSAM/runs/segment/pupil_retrain/weights/best.onnx'
VAL_DIR   = '/home/ls5255/FastSAM/dataset_pupil_seg/images/val'
OUT_DIR   = '/home/ls5255/.gemini/antigravity/brain/a44cb83f-9fff-4bdd-8a5a-1ba663634d74'

files = [os.path.join(VAL_DIR, f) for f in sorted(os.listdir(VAL_DIR)) if f.endswith(('.png','.jpg'))][:9]

def run_grid(device_str, label):
    print(f"\n--- Running ONNX INT8 on {label} ({device_str}) ---")
    model = YOLO(ONNX_PATH, task='segment')
    imgs = []
    total_ms = 0
    for i, f in enumerate(files):
        t0 = time.perf_counter()
        res = model.predict(source=f, imgsz=1024, device=device_str, retina_masks=True, verbose=False)
        elapsed = (time.perf_counter() - t0) * 1000
        total_ms += elapsed
        plotted = res[0].plot(boxes=False, labels=False, conf=False)
        plotted = cv2.resize(plotted, (640, 240))
        # Add label overlay
        cv2.putText(plotted, f'{label} | {elapsed:.0f}ms', (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
        imgs.append(plotted)
        print(f"  [{i+1}/9] {os.path.basename(f)} -> {elapsed:.1f}ms")
    
    avg = total_ms / len(files)
    print(f"  Average: {avg:.1f}ms per image")
    
    h1 = cv2.hconcat(imgs[0:3])
    h2 = cv2.hconcat(imgs[3:6])
    h3 = cv2.hconcat(imgs[6:9])
    return cv2.vconcat([h1, h2, h3]), avg

# GPU grid
gpu_grid, gpu_avg = run_grid('0', 'ONNX INT8 GPU')
cv2.imwrite(os.path.join(OUT_DIR, 'onnx_int8_val_gpu_grid.jpg'), gpu_grid)
print(f"Saved GPU grid")

# CPU grid
cpu_grid, cpu_avg = run_grid('cpu', 'ONNX INT8 CPU')
cv2.imwrite(os.path.join(OUT_DIR, 'onnx_int8_val_cpu_grid.jpg'), cpu_grid)
print(f"Saved CPU grid")

print(f"\nDone! GPU avg: {gpu_avg:.1f}ms, CPU avg: {cpu_avg:.1f}ms")
