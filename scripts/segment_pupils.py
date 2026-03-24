"""
Pupil Segmentation using FastSAM

This script segments pupils from eye-tracking images using FastSAM,
then filters the results to isolate only the pupil regions.
Optionally fits ellipses for cleaner boundaries.
"""

import os
import sys
import argparse
import numpy as np
import cv2
from PIL import Image
import torch
from pathlib import Path
import urllib.request

# Add FastSAM to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastsam import FastSAM, FastSAMPrompt


def download_weights(weights_path):
    """Download FastSAM weights if not present."""
    if os.path.exists(weights_path):
        return True
    
    os.makedirs(os.path.dirname(weights_path), exist_ok=True)
    url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/FastSAM-x.pt"
    
    print(f"Downloading FastSAM weights from {url}...")
    try:
        urllib.request.urlretrieve(url, weights_path)
        print(f"Downloaded weights to {weights_path}")
        return True
    except Exception as e:
        print(f"Error downloading weights: {e}")
        return False


def get_mask_properties(mask):
    """Calculate properties of a binary mask."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return None
    
    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)
    
    if area < 10:
        return None
    
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    M = cv2.moments(largest_contour)
    if M["m00"] == 0:
        return None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    
    perimeter = cv2.arcLength(largest_contour, True)
    circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
    
    ellipse = None
    aspect_ratio = w / h if h > 0 else 0
    if len(largest_contour) >= 5:
        try:
            ellipse = cv2.fitEllipse(largest_contour)
            (ex, ey), (ma, MA), angle = ellipse
            aspect_ratio = min(ma, MA) / max(ma, MA) if max(ma, MA) > 0 else 0
        except:
            pass
    
    return {
        'area': area,
        'centroid': (cx, cy),
        'bbox': (x, y, w, h),
        'circularity': circularity,
        'aspect_ratio': aspect_ratio,
        'contour': largest_contour,
        'ellipse': ellipse,
        'width': w,
        'height': h
    }


def calculate_mean_intensity(image, mask):
    """Calculate mean intensity of image within mask region."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    masked_pixels = gray[mask > 0]
    return np.mean(masked_pixels) if len(masked_pixels) > 0 else 255


def calculate_min_intensity(image, mask):
    """Calculate minimum intensity of image within mask region."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    masked_pixels = gray[mask > 0]
    return np.min(masked_pixels) if len(masked_pixels) > 0 else 255


def create_ellipse_mask(ellipse, img_shape):
    """Create a binary mask from an ellipse."""
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    cv2.ellipse(mask, ellipse, 255, -1)
    return mask > 0


def is_pupil_candidate(props, image, mask, img_width, img_height, side='left', debug=False):
    """
    Determine if a mask is likely a pupil based on various criteria.
    
    Key filtering based on IR eye image properties:
    - Pupils are DARK (low intensity in IR)
    - Pupils are OVAL-shaped (not necessarily circular)
    - Pupils have a specific size range (typically 500-15000 pixels)
    - Pupils are located in expected regions (left half for left eye, right half for right eye)
    """
    if props is None:
        return False, 0
    
    cx, cy = props['centroid']
    area = props['area']
    circularity = props['circularity']
    aspect_ratio = props['aspect_ratio']
    width = props['width']
    height = props['height']
    bbox_x, bbox_y, bbox_w, bbox_h = props['bbox']
    
    mean_intensity = calculate_mean_intensity(image, mask)
    min_intensity = calculate_min_intensity(image, mask)
    
    # ABSOLUTE pixel area thresholds based on actual pupil measurements
    # Observed range: 914-5963 pixels, mean ~3500
    # Using wider range to handle variation: 500-15000 pixels
    MIN_PUPIL_AREA = 500
    MAX_PUPIL_AREA = 15000
    
    # Maximum dimension - be generous since pupils can appear large depending on viewing angle
    max_dim = min(img_width, img_height) * 0.35  # Allow up to 35% of smaller dimension
    
    mid_x = img_width // 2
    quarter_x = img_width // 4
    
    # More permissive edge margins
    edge_margin_x = img_width * 0.03
    edge_margin_y = img_height * 0.05
    
    if side == 'left':
        in_correct_region = (edge_margin_x < cx < mid_x - edge_margin_x)
        position_quality = 1.0 - abs(cx - quarter_x) / quarter_x
    else:
        in_correct_region = (mid_x + edge_margin_x < cx < img_width - edge_margin_x)
        position_quality = 1.0 - abs(cx - 3*quarter_x) / quarter_x
    
    position_quality = max(0, min(1, position_quality))
    
    # Check if CENTROID is too close to image edge
    too_close_to_edge = (
        cx < edge_margin_x or 
        cx > img_width - edge_margin_x or
        cy < edge_margin_y or 
        cy > img_height - edge_margin_y
    )
    
    score = 0
    
    # ============================================
    # DARKNESS SCORE (most important for IR pupils)
    # ============================================
    # Pupils are the darkest region in IR images
    if mean_intensity < 20:
        score += 50
    elif mean_intensity < 30:
        score += 40
    elif mean_intensity < 45:
        score += 30
    elif mean_intensity < 60:
        score += 15
    elif mean_intensity < 80:
        score += 5
    else:
        score -= 30  # Too bright to be a pupil
    
    # Bonus for very dark minimum intensity
    if min_intensity < 10:
        score += 15
    elif min_intensity < 20:
        score += 10
    elif min_intensity < 30:
        score += 5
    
    # ============================================
    # SHAPE SCORE (pupils are oval, NOT necessarily circular)
    # ============================================
    # Observed circularity range: 0.42-0.75
    # We should NOT penalize low circularity since pupils are oval
    # Only give small bonus for more circular shapes
    if circularity > 0.6:
        score += 5  # Small bonus for rounder shapes
    elif circularity > 0.35:
        score += 3  # Acceptable oval shape
    # No penalty for low circularity - ovals are expected!
    
    # Aspect ratio - pupils can be quite elongated
    # Only penalize extreme elongation
    if aspect_ratio > 0.3:
        score += 5  # Reasonable aspect ratio
    elif aspect_ratio < 0.2:
        score -= 10  # Extremely elongated, unlikely to be pupil
    
    # ============================================
    # SIZE SCORE (absolute pixel area)
    # ============================================
    if MIN_PUPIL_AREA <= area <= MAX_PUPIL_AREA:
        score += 20  # Good size range
    elif area < MIN_PUPIL_AREA * 0.5:
        score -= 20  # Too small
    elif area > MAX_PUPIL_AREA * 1.5:
        score -= 25  # Too large
    elif area < MIN_PUPIL_AREA:
        score -= 5   # Slightly small
    elif area > MAX_PUPIL_AREA:
        score -= 10  # Slightly large
    
    # ============================================
    # POSITION SCORE
    # ============================================
    score += int(position_quality * 15)
    
    # Edge penalty
    if too_close_to_edge:
        score -= 30
    
    # Region check - must be in correct half
    if not in_correct_region:
        score -= 100
    
    # ============================================
    # FINAL CANDIDATE CHECK
    # ============================================
    is_candidate = (
        score >= 35 and
        in_correct_region and 
        not too_close_to_edge and
        MIN_PUPIL_AREA * 0.3 <= area <= MAX_PUPIL_AREA * 2 and
        mean_intensity < 100 and
        width < max_dim and 
        height < max_dim
    )
    
    if debug and (is_candidate or score > 25):
        edge_str = " [EDGE]" if too_close_to_edge else ""
        region_str = "" if in_correct_region else " [WRONG REGION]"
        print(f"    Mask ({side}): cx={cx}, cy={cy}, area={area:.0f}, "
              f"circ={circularity:.2f}, int={mean_intensity:.1f}, score={score}{edge_str}{region_str}")
    
    return is_candidate, score


def segment_pupils(image_path, model, device='cuda', output_path=None, 
                  conf=0.25, iou=0.9, imgsz=1024, debug=False, use_ellipse=True,
                  resize_scale=1.0, use_half=False, viz_style='default'):
    """
    Segment pupils from an eye image.
    
    Args:
        use_ellipse: If True, fit ellipses to masks for cleaner boundaries
        resize_scale: Scale factor for input before inference (e.g. 0.5 = faster, less accurate)
        use_half: Use FP16 for model and input (faster on GPU, may be less stable)
        viz_style: 'default' (teal/brown masks, X crosshairs) or 'paper'
            (red/green masks, + crosshairs — matches common IR pupil-segmentation figures).
    """
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)
    img_height, img_width = image_np.shape[:2]

    # Optional resize for speed (inference on smaller image, then scale back)
    infer_image = image
    scale_x, scale_y = 1.0, 1.0
    if resize_scale is not None and resize_scale < 1.0 and resize_scale > 0:
        new_w = max(64, int(img_width * resize_scale))
        new_h = max(64, int(img_height * resize_scale))
        infer_image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        scale_x = img_width / new_w
        scale_y = img_height / new_h
        if debug:
            print(f"  Resized for inference: {new_w}x{new_h} (scale back by {scale_x:.2f}x{scale_y:.2f})")

    if debug:
        print(f"  Image size: {img_width}x{img_height}")
    
    # Use inference_mode to speed up and reduce memory
    with torch.inference_mode():
        everything_results = model(
            infer_image,
            device=device,
            retina_masks=True,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
        )
    
    if everything_results is None or len(everything_results) == 0:
        if debug:
            print(f"No segments found in {image_path}")
        return None
    
    prompt_process = FastSAMPrompt(infer_image, everything_results, device=device)
    all_masks = prompt_process.everything_prompt()
    
    if len(all_masks) == 0:
        if debug:
            print(f"No masks generated for {image_path}")
        return None
    
    if isinstance(all_masks, torch.Tensor):
        all_masks = all_masks.cpu().numpy()
    
    if debug:
        print(f"  Total masks from FastSAM: {len(all_masks)}")
    
    left_candidates = []
    right_candidates = []
    
    infer_h, infer_w = np.array(infer_image).shape[:2]
    for i, mask in enumerate(all_masks):
        mask_h, mask_w = mask.shape[:2]
        if mask_h != img_height or mask_w != img_width:
            # Resize mask to original image size (scale back if we resized for inference)
            mask = cv2.resize(mask.astype(np.float32), (img_width, img_height), 
                            interpolation=cv2.INTER_NEAREST)
        mask = mask > 0.5
        
        props = get_mask_properties(mask)
        if props is None:
            continue
        
        is_left, left_score = is_pupil_candidate(
            props, image_np, mask, img_width, img_height, side="left", debug=debug
        )
        if is_left:
            mean_int = calculate_mean_intensity(image_np, mask)
            left_candidates.append({
                'mask': mask,
                'props': props,
                'score': left_score,
                'intensity': mean_int,
                'index': i
            })
        
        is_right, right_score = is_pupil_candidate(
            props, image_np, mask, img_width, img_height, side='right', debug=debug
        )
        if is_right:
            mean_int = calculate_mean_intensity(image_np, mask)
            right_candidates.append({
                'mask': mask,
                'props': props,
                'score': right_score,
                'intensity': mean_int,
                'index': i
            })
    
    if debug:
        print(f"  Valid candidates - Left: {len(left_candidates)}, Right: {len(right_candidates)}")
    
    left_pupil = None
    right_pupil = None
    
    if left_candidates:
        left_candidates.sort(key=lambda x: (-x['score'], x['intensity']))
        left_pupil = left_candidates[0]
        
        # Fit ellipse for cleaner mask
        if use_ellipse and left_pupil['props']['ellipse'] is not None:
            ellipse = left_pupil['props']['ellipse']
            left_pupil['ellipse_mask'] = create_ellipse_mask(ellipse, (img_height, img_width))
        
        if debug:
            cx, cy = left_pupil['props']['centroid']
            print(f"  Selected LEFT: score={left_pupil['score']}, intensity={left_pupil['intensity']:.1f}, "
                  f"center=({cx}, {cy})")
    
    if right_candidates:
        right_candidates.sort(key=lambda x: (-x['score'], x['intensity']))
        right_pupil = right_candidates[0]
        
        if use_ellipse and right_pupil['props']['ellipse'] is not None:
            ellipse = right_pupil['props']['ellipse']
            right_pupil['ellipse_mask'] = create_ellipse_mask(ellipse, (img_height, img_width))
        
        if debug:
            cx, cy = right_pupil['props']['centroid']
            print(f"  Selected RIGHT: score={right_pupil['score']}, intensity={right_pupil['intensity']:.1f}, "
                  f"center=({cx}, {cy})")
    
    if output_path:
        output_img = image_np.copy()
        overlay = output_img.copy()
        
        # Colors (RGB format)
        if viz_style == 'paper':
            left_color = (255, 0, 0)      # Red (left eye in figure convention)
            right_color = (0, 255, 0)     # Green (right eye)
            crosshair_color_left = (255, 40, 40)    # Bright red
            crosshair_color_right = (40, 255, 40)   # Bright green
            use_plus_crosshair = True
        else:
            left_color = (0, 128, 128)    # Teal
            right_color = (160, 128, 96)  # Brown
            crosshair_color_left = (0, 255, 255)    # Cyan
            crosshair_color_right = (255, 165, 0)   # Orange
            use_plus_crosshair = False
        
        left_center = None
        right_center = None
        
        if left_pupil is not None:
            # Use ellipse mask if available, otherwise use original mask
            if use_ellipse and 'ellipse_mask' in left_pupil:
                mask = left_pupil['ellipse_mask']
            else:
                mask = left_pupil['mask']
            overlay[mask > 0] = left_color
            left_center = left_pupil['props']['centroid']
        
        if right_pupil is not None:
            if use_ellipse and 'ellipse_mask' in right_pupil:
                mask = right_pupil['ellipse_mask']
            else:
                mask = right_pupil['mask']
            overlay[mask > 0] = right_color
            right_center = right_pupil['props']['centroid']
        
        alpha = 0.7
        result = cv2.addWeighted(overlay, alpha, output_img, 1 - alpha, 0)
        
        # Draw crosshairs at pupil centers
        crosshair_size = 15  # pixels
        crosshair_thickness = 2
        
        def draw_crosshair(img, cx, cy, color, plus):
            if plus:
                cv2.line(img, (cx - crosshair_size, cy), (cx + crosshair_size, cy),
                         color, crosshair_thickness)
                cv2.line(img, (cx, cy - crosshair_size), (cx, cy + crosshair_size),
                         color, crosshair_thickness)
            else:
                cv2.line(img, (cx - crosshair_size, cy - crosshair_size),
                         (cx + crosshair_size, cy + crosshair_size), color, crosshair_thickness)
                cv2.line(img, (cx - crosshair_size, cy + crosshair_size),
                         (cx + crosshair_size, cy - crosshair_size), color, crosshair_thickness)
            cv2.circle(img, (cx, cy), 3, color, -1)
        
        if left_center is not None:
            cx, cy = int(left_center[0]), int(left_center[1])
            draw_crosshair(result, cx, cy, crosshair_color_left, use_plus_crosshair)
        
        if right_center is not None:
            cx, cy = int(right_center[0]), int(right_center[1])
            draw_crosshair(result, cx, cy, crosshair_color_right, use_plus_crosshair)
        
        result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        cv2.imwrite(output_path, result_bgr)
        print(f"Saved: {output_path}")
    
    # Extract pupil centers for easy access
    left_center = left_pupil['props']['centroid'] if left_pupil else None
    right_center = right_pupil['props']['centroid'] if right_pupil else None
    
    return {
        'left': left_pupil,
        'right': right_pupil,
        'left_center': left_center,   # (x, y) tuple or None
        'right_center': right_center, # (x, y) tuple or None
        'all_masks_count': len(all_masks)
    }


def process_directory(input_dir, output_dir, model, device='cuda', 
                      conf=0.25, iou=0.9, imgsz=1024, debug=False, use_ellipse=True,
                      resize_scale=1.0, use_half=False, viz_style='default'):
    """Process all images in a directory."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    image_files = [f for f in input_path.iterdir() 
                   if f.suffix.lower() in image_extensions]
    
    print(f"Found {len(image_files)} images to process")
    
    results = {}
    for i, img_file in enumerate(sorted(image_files)):
        print(f"\nProcessing [{i+1}/{len(image_files)}]: {img_file.name}")
        
        output_file = output_path / img_file.name
        
        result = segment_pupils(
            str(img_file), 
            model, 
            device=device,
            output_path=str(output_file),
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            debug=debug,
            use_ellipse=use_ellipse,
            resize_scale=resize_scale,
            use_half=use_half,
        )
        
        if result:
            results[img_file.name] = {
                'left_found': result['left'] is not None,
                'right_found': result['right'] is not None,
                'total_masks': result['all_masks_count']
            }
    
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    total = len(results)
    if total > 0:
        left_found = sum(1 for r in results.values() if r['left_found'])
        right_found = sum(1 for r in results.values() if r['right_found'])
        both_found = sum(1 for r in results.values() if r['left_found'] and r['right_found'])
        
        print(f"Total images processed: {total}")
        print(f"Left pupil found: {left_found}/{total} ({100*left_found/total:.1f}%)")
        print(f"Right pupil found: {right_found}/{total} ({100*right_found/total:.1f}%)")
        print(f"Both pupils found: {both_found}/{total} ({100*both_found/total:.1f}%)")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Segment pupils using FastSAM')
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Input image or directory')
    parser.add_argument('--output', '-o', type=str, default='./output_pupils/',
                        help='Output directory')
    parser.add_argument('--model_path', type=str, default='./weights/FastSAM-x.pt',
                        help='Path to FastSAM weights')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda/cpu/mps)')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.9,
                        help='IOU threshold')
    parser.add_argument('--imgsz', type=int, default=1024,
                        help='Image size for inference (e.g. 512 for faster, 1024 for accuracy)')
    parser.add_argument('--resize', type=float, default=1.0,
                        help='Resize input by this factor before inference (e.g. 0.5 = half size, faster)')
    parser.add_argument('--half', action='store_true',
                        help='Use FP16 for model and input (faster on GPU, may be less stable)')
    parser.add_argument('--quantize', action='store_true',
                        help='Apply dynamic INT8 quantization to FastSAM head (CPU only, experimental)')
    parser.add_argument('--compile', action='store_true',
                        help='Use torch.compile on the FastSAM model (PyTorch 2+, experimental)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug output')
    parser.add_argument('--no-ellipse', action='store_true',
                        help='Disable ellipse fitting (use raw mask)')
    parser.add_argument(
        '--viz-style',
        type=str,
        default='default',
        choices=('default', 'paper'),
        help="Overlay style: 'default' (teal/brown + X) or 'paper' (red/green masks + + crosshairs).",
    )
    
    args = parser.parse_args()
    
    if args.device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    else:
        device = args.device
    
    print(f"Using device: {device}")
    use_half = getattr(args, 'half', False)
    if use_half and device == 'cuda':
        print("FP16 (half) enabled for faster inference.")
    
    if not download_weights(args.model_path):
        return
    
    print(f"Loading FastSAM model from {args.model_path}...")
    model = FastSAM(args.model_path)

    # Optional FP16
    if use_half and device == 'cuda':
        try:
            if hasattr(model, 'model'):
                model.model = model.model.half()
        except Exception as e:
            print(f"Warning: could not set half precision: {e}")

    # Optional dynamic post-training quantization (CPU only)
    if args.quantize:
        if device != 'cpu':
            print("Quantization is CPU-only; switching device to 'cpu' for inference.")
            device = 'cpu'
        try:
            from torch.ao.quantization import quantize_dynamic
            if hasattr(model, 'model'):
                print("Applying dynamic INT8 quantization to FastSAM backbone/head (Linear layers).")
                model.model = quantize_dynamic(
                    model.model,
                    {torch.nn.Linear},
                    dtype=torch.qint8,
                )
            else:
                print("Warning: FastSAM object has no 'model' attribute; skipping quantization.")
        except Exception as e:
            print(f"Warning: dynamic quantization failed (will continue without it): {e}")

    # Optional torch.compile for further speedup (PyTorch 2+)
    if getattr(args, 'compile', False):
        if hasattr(torch, 'compile'):
            try:
                if hasattr(model, 'model'):
                    print("Compiling FastSAM model with torch.compile (this may take a while on first run)...")
                    model.model = torch.compile(model.model)
                else:
                    print("Warning: FastSAM object has no 'model' attribute; skipping compile.")
            except Exception as e:
                print(f"Warning: torch.compile failed (will continue without it): {e}")
        else:
            print("torch.compile not available in this PyTorch version; ignoring --compile.")
    
    input_path = Path(args.input)
    use_ellipse = not args.no_ellipse
    
    resize_scale = getattr(args, 'resize', 1.0)
    use_half = getattr(args, 'half', False)
    if input_path.is_dir():
        process_directory(
            args.input, args.output, model, device=device,
            conf=args.conf, iou=args.iou, imgsz=args.imgsz, debug=args.debug,
            use_ellipse=use_ellipse, resize_scale=resize_scale, use_half=use_half,
            viz_style=args.viz_style,
        )
    else:
        output_file = Path(args.output) / input_path.name
        segment_pupils(
            args.input, model, device=device, output_path=str(output_file),
            conf=args.conf, iou=args.iou, imgsz=args.imgsz, debug=args.debug,
            use_ellipse=use_ellipse, resize_scale=resize_scale, use_half=use_half,
            viz_style=args.viz_style,
        )


if __name__ == '__main__':
    main()
