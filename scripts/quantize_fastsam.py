#!/usr/bin/env python3
"""
Export or post-train quantize FastSAM (YOLOv8-segmentation) weights.

This targets **this repo’s CNN segment model**, not Meta SAM (see QUANTIZATION.md for PTQ4SAM).

Modes:
  onnx / onnx_half — Ultralytics ONNX export (FP32 / FP16 weights in graph where supported)
  engine           — TensorRT .engine (GPU, FP16 when half=True)
  torchscript      — TorchScript
  dynamic_ptq      — PyTorch dynamic INT8 on Linear/Conv (CPU-friendly; custom load path)

For INT8 ONNX/TensorRT calibration in some toolchains, pass --data pointing to data.yaml and
use images under dataset_pupil_seg/images/train as an unlabeled calibration set (same role as
PTQ calibration images).

Examples:
  python quantize_fastsam.py --weights runs/segment/pupil_retrain/weights/best.pt --mode onnx_half --imgsz 1024 --device 0
  python quantize_fastsam.py --weights weights/FastSAM-x.pt --mode onnx --output exports/
  python quantize_fastsam.py --weights best.pt --mode dynamic_ptq --output weights/best_dynamic_int8.pt
  python quantize_fastsam.py --weights best.pt --mode engine --imgsz 1024 --device 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastsam import FastSAM


def collect_calib_images(calib_dir: Path | None, max_images: int) -> list[Path]:
    if calib_dir is None or not calib_dir.is_dir():
        return []
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    files: list[Path] = []
    for p in sorted(calib_dir.rglob("*")):
        if p.suffix.lower() in exts:
            files.append(p)
            if len(files) >= max_images:
                break
    return files


def main() -> None:
    ap = argparse.ArgumentParser(description="Export / PTQ helper for FastSAM .pt weights")
    ap.add_argument(
        "--weights",
        "-w",
        type=str,
        default="weights/FastSAM-x.pt",
        help="Path to checkpoint (.pt), e.g. runs/segment/pupil_retrain/weights/best.pt",
    )
    ap.add_argument(
        "--output",
        "-o",
        type=str,
        default="",
        help="Output directory for exports (default: same folder as weights)",
    )
    ap.add_argument(
        "--mode",
        type=str,
        choices=("onnx", "onnx_half", "engine", "torchscript", "dynamic_ptq"),
        default="onnx_half",
        help="Export or quantization mode",
    )
    ap.add_argument("--imgsz", type=int, default=1024, help="Square input size (match training/inference)")
    ap.add_argument(
        "--device",
        type=str,
        default=None,
        help="cuda device index or 'cpu'. FP16 ONNX/engine need GPU in this Ultralytics build.",
    )
    ap.add_argument("--simplify", action="store_true", default=True, help="ONNX simplify (default on)")
    ap.add_argument("--no-simplify", action="store_false", dest="simplify", help="Disable ONNX simplify")
    ap.add_argument(
        "--data",
        type=str,
        default=None,
        help="Optional dataset/data.yaml (metadata for export logs; required for some TF INT8 paths)",
    )
    ap.add_argument(
        "--calib-dir",
        type=str,
        default=None,
        help="Folder of images (e.g. dataset_pupil_seg/images/train) for documentation / future calibrators",
    )
    ap.add_argument("--max-calib", type=int, default=32, help="Max images to list from --calib-dir")
    args = ap.parse_args()

    weights_path = Path(args.weights).resolve()
    if not weights_path.exists():
        print(f"ERROR: weights not found: {weights_path}")
        sys.exit(1)

    out_dir = Path(args.output).resolve() if args.output else weights_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    calib_list = collect_calib_images(Path(args.calib_dir).resolve() if args.calib_dir else None, args.max_calib)
    if calib_list:
        print(f"Calibration images listed ({len(calib_list)}): {calib_list[0].parent} ...")
    elif args.calib_dir:
        print(f"WARNING: no images found under --calib-dir {args.calib_dir}")

    device = args.device
    if device is None:
        device = "0" if torch.cuda.is_available() else "cpu"

    half = args.mode in ("onnx_half", "engine")
    fmt_map = {
        "onnx": "onnx",
        "onnx_half": "onnx",
        "engine": "engine",
        "torchscript": "torchscript",
    }

    print(f"Loading {weights_path} ...")
    model = FastSAM(str(weights_path))

    if args.mode == "dynamic_ptq":
        print("Applying torch.quantization.quantize_dynamic (Linear + Conv2d) ...")
        pt_model = model.model
        pt_model.eval()
        quantized = torch.quantization.quantize_dynamic(
            pt_model, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8
        )
        out_pt = out_dir / f"{weights_path.stem}_dynamic_int8.pt"
        torch.save(quantized.state_dict(), out_pt)
        print(f"Saved dynamic-quantized state_dict to {out_pt}")
        print(
            "NOTE: Loading this state_dict requires building the same nn.Module and load_state_dict; "
            "for deployment prefer ONNX/engine export."
        )
        return

    export_kw: dict = {
        "format": fmt_map[args.mode],
        "imgsz": args.imgsz,
        "simplify": args.simplify,
        "device": device,
        "half": half,
    }
    if args.data:
        export_kw["data"] = str(Path(args.data).resolve())

    # Export writes next to checkpoint by default; copy hint
    print(f"Exporting mode={args.mode} -> {export_kw['format']} half={half} imgsz={args.imgsz} device={device}")
    try:
        paths = model.export(**export_kw)
    except AssertionError as e:
        print(f"ERROR: {e}")
        print("Hint: use --device 0 for half/engine export, or use --mode onnx (FP32) on CPU.")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: export failed: {e}")
        raise

    print(f"Export finished: {paths}")
    print("Validate quality: python benchmark_inference.py --weights {w} --data dataset_pupil_seg/data.yaml --val-only".format(w=weights_path))
    print("Benchmark speed:   python benchmark_inference.py --weights {w} --imgsz {i} [--half]".format(w=weights_path, i=args.imgsz))


if __name__ == "__main__":
    main()
