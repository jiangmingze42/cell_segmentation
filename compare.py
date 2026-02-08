#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import cv2
import tifffile as tiff
from PIL import Image



IN_TIF = "/home/student002/MIT_segmetation/real_rgb/2024-10-22_223241_leg1-1.tif"         
IMG_DIR = "/home/student002/MIT_segmetation/sam3/slam_seg/output/composite_rgb_new/2024-10-22_223241_leg1_rgb_frames"       
OUT_DIR = "/home/student002/MIT_segmetation/cell_segmentation/outputs/2024-10-22_223241_leg1_1008"       

OUT_TIF = "compare/2024-10-22_223241_leg1.tif"   
OUT_PNG_DIR = "compare/compare_png/2024-10-22_182459_skin1"       

SAVE_FRAMES = [0, 1, 5, 10, 11, 12]

# 只处理前 N 帧（0=全处理）
LIMIT = 0


LABEL_LEFT = "original"
LABEL_MID = "processed"
LABEL_RIGHT = "result"

FONT_SCALE = 0.7


def tif_to_TCYX(arr: np.ndarray) -> np.ndarray:
    """
    Accept TIF arrays in one of:
      - (T,3,H,W)
      - (T,H,W,3)
      - (3,H,W)
      - (H,W,3)
    Return unified: (T,3,H,W)
    """
    a = np.asarray(arr)

    if a.ndim == 4:
        if a.shape[1] == 3:          # (T,3,H,W)
            return a
        if a.shape[-1] == 3:         # (T,H,W,3)
            return np.transpose(a, (0, 3, 1, 2))
        raise ValueError(f"Unsupported 4D tif shape {a.shape}. Need (T,3,H,W) or (T,H,W,3).")

    if a.ndim == 3:
        if a.shape[0] == 3:          # (3,H,W)
            return a[None, ...]
        if a.shape[-1] == 3:         # (H,W,3)
            a2 = np.transpose(a, (2, 0, 1))
            return a2[None, ...]
        raise ValueError(f"Unsupported 3D tif shape {a.shape}. Need (3,H,W) or (H,W,3).")

    raise ValueError(f"Unsupported tif ndim={a.ndim}, shape={a.shape}")


def to_rgb_u8_from_CYX(frame_cyx: np.ndarray) -> np.ndarray:
    """(3,H,W) -> (H,W,3) uint8"""
    rgb = np.transpose(frame_cyx, (1, 2, 0))
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return rgb


def resize_to_hw(img_rgb_u8: np.ndarray, H: int, W: int) -> np.ndarray:
    if img_rgb_u8.shape[0] == H and img_rgb_u8.shape[1] == W:
        return img_rgb_u8
    return np.array(Image.fromarray(img_rgb_u8).resize((W, H), Image.BILINEAR), dtype=np.uint8)


def put_label(img_rgb_u8: np.ndarray, text: str, x: int, y: int = 24,
              font_scale: float = 0.7, thickness: int = 2) -> np.ndarray:
    bgr = cv2.cvtColor(img_rgb_u8, cv2.COLOR_RGB2BGR)
    font = cv2.FONT_HERSHEY_SIMPLEX

    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    pad = 6

    x1, y1 = x, max(0, y - th - pad)
    x2 = min(bgr.shape[1] - 1, x + tw + 2 * pad)
    y2 = min(bgr.shape[0] - 1, y + baseline + pad)

    cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 0, 0), -1)
    cv2.putText(bgr, text, (x + pad, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def make_triptych(left_rgb: np.ndarray, mid_rgb: np.ndarray, right_rgb: np.ndarray,
                  label_left: str, label_mid: str, label_right: str,
                  frame_idx: int, font_scale: float = 0.7) -> np.ndarray:
    H, W, _ = left_rgb.shape
    mid_rgb = resize_to_hw(mid_rgb, H, W)
    right_rgb = resize_to_hw(right_rgb, H, W)

    canvas = np.concatenate([left_rgb, mid_rgb, right_rgb], axis=1)

    canvas = put_label(canvas, f"{label_left}", x=10, y=24, font_scale=font_scale)
    canvas = put_label(canvas, f"{label_mid}", x=10 + W, y=24, font_scale=font_scale)
    canvas = put_label(canvas, f"{label_right}", x=10 + 2 * W, y=24, font_scale=font_scale)
    return canvas


def main():
    in_tif = Path(IN_TIF)
    img_dir = Path(IMG_DIR)
    out_dir = Path(OUT_DIR)

    overlays_dir = out_dir / "overlays_png"
    if not overlays_dir.exists():
        raise FileNotFoundError(f"Cannot find overlays_png: {overlays_dir} (need output from your first program)")

    out_png_dir = Path(OUT_PNG_DIR)
    out_png_dir.mkdir(parents=True, exist_ok=True)

    out_tif = Path(OUT_TIF) if OUT_TIF else None
    if out_tif is not None:
        out_tif.parent.mkdir(parents=True, exist_ok=True)

    raw = tiff.imread(str(in_tif))
    arr = tif_to_TCYX(raw)
    T, C, H, W = arr.shape
    if C != 3:
        raise ValueError(f"After conversion, expect C=3, got C={C}")

    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
    images = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in exts])
    if len(images) == 0:
        raise ValueError(f"No images found in {img_dir}")

    N = min(len(images), T)
    if LIMIT and LIMIT > 0:
        N = min(N, LIMIT)

    save_set = set(SAVE_FRAMES)

    trip_frames = [] 

    for t in range(N):
        left = to_rgb_u8_from_CYX(arr[t])

        mid = np.array(Image.open(images[t]).convert("RGB"), dtype=np.uint8)
        mid = resize_to_hw(mid, H, W)

        ov_path = overlays_dir / f"t{t:03d}_overlay.png"
        if not ov_path.exists():
            raise FileNotFoundError(f"Missing overlay for t={t}: {ov_path}")

        right = np.array(Image.open(ov_path).convert("RGB"), dtype=np.uint8)
        right = resize_to_hw(right, H, W)

        trip = make_triptych(
            left, mid, right,
            LABEL_LEFT, LABEL_MID, LABEL_RIGHT,
            frame_idx=t,
            font_scale=FONT_SCALE,
        )

        if t in save_set:
            Image.fromarray(trip).save(out_png_dir / f"t{t:03d}_triptych.png")

        if out_tif is not None:
            trip_frames.append(trip)

        if (t + 1) % 50 == 0 or t == N - 1:
            print(f"[INFO] processed {t+1}/{N}")

    if out_tif is not None and len(trip_frames) > 0:
        trip_stack = np.stack(trip_frames, axis=0)  # (N,H,3W,3)
        tiff.imwrite(
            str(out_tif),
            trip_stack,
            photometric="rgb",
            bigtiff=True,
            compression="deflate",
        )
        print("[OK] out tif ->", out_tif)

    print("[OK] saved png frames ->", out_png_dir)


if __name__ == "__main__":
    main()
