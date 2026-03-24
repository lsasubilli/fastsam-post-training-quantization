#!/usr/bin/env python3
import os
import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    print("Install ultralytics")
    exit(1)

# Path to the ACTUAL ONNX INT8 PTQ MODEL
model_path = '/home/ls5255/FastSAM/runs/segment/pupil_retrain/weights/best.onnx'
val_dir = '/home/ls5255/FastSAM/dataset_pupil_seg/images/val'
out_dir = '/home/ls5255/FastSAM/paper_figures_inference/real_ptq_proof/ptq_val_grid'
os.makedirs(out_dir, exist_ok=True)

print("Loading INT8 PTQ model...")
model = YOLO(model_path, task='segment')

files = [os.path.join(val_dir, f) for f in sorted(os.listdir(val_dir)) if f.endswith(('.png', '.jpg'))][:9]

imgs = []
print("Running PTQ inference...")
for i, f in enumerate(files):
    # Predict without boxes or labels in plot()
    res = model.predict(source=f, imgsz=1024, device='0', retina_masks=True, verbose=False)
    # Save the output with boxes=False and labels=False
    res_plotted = res[0].plot(boxes=False, labels=False, conf=False)
    
    # Resize to standard width/height to make grid creation safe
    res_plotted = cv2.resize(res_plotted, (1280, 480))
    imgs.append(res_plotted)
    print(f"Processed {f} (INT8 PTQ)")

if len(imgs) == 9:
    h1 = cv2.hconcat(imgs[0:3])
    h2 = cv2.hconcat(imgs[3:6])
    h3 = cv2.hconcat(imgs[6:9])
    grid = cv2.vconcat([h1, h2, h3])
    out_path = '/home/ls5255/.gemini/antigravity/brain/a44cb83f-9fff-4bdd-8a5a-1ba663634d74/ptq_int8_predictions_clean_grid.jpg'
    cv2.imwrite(out_path, grid)
    print('Saved PTQ INT8 Grid:', out_path)
