#!/usr/bin/env python3
import os
from pathlib import Path

import numpy as np
import tifffile as tiff


# =========================
# Inputs
# =========================
RAW_TIFS = [
    "/home/student002/MIT_segmetation/lyme-disease-images/2024-10-22_182459_skin1.tif",
    "/home/student002/MIT_segmetation/lyme-disease-images/2024-10-23_151912_skin1.tif",
    "/home/student002/MIT_segmetation/lyme-disease-images/2024-11-05_182444_Z_300um_6us_tail2.tif"
]


OUT_DIR = Path("preprocessed_rgb")


# =========================
# Global normalization config (用于从 raw 生成 rgb)
# =========================
RAW_P_LOW = 1.0
RAW_P_HIGH = 99.8
RAW_GAMMA = 1.0

# 通道：0(THG),1(NADH),2(SHG),3(FAD)
KEEP_CH = [0, 1, 2, 3]


# =========================
# Composition config（按你给的程序）
# =========================
SHG_SCALE = 0.0            # G通道里 SHG 的权重
SHG_PRESENT_THR = 0.0      # shg > thr 判定“绿色存在”
THG_MINUS_SHG_SCALE = 2.0  # thg_corr = thg - scale*shg


# =========================
# Frame export config（切片输出 png）
# =========================
EXPORT_FORMAT = "png"   # "png" 或 "tif" 都可以
EXPORT_PAD = 6          # frame_000000.png


ENABLE_FRAME_CONTRAST_STRETCH = True
FRAME_P_LOW = 0.3
FRAME_P_HIGH = 99.8
FRAME_GAMMA = None      # None or float, e.g. 0.8 / 1.2


# =========================
# Helpers
# =========================
def compute_global_lo_hi_selected(arr: np.ndarray, ch_idx, p_low: float, p_high: float):
    """
    arr: (T,C,H,W) or (C,H,W)
    return dict: {c: (lo, hi)}
    """
    stats = {}
    if arr.ndim == 4:
        for c in ch_idx:
            x = arr[:, c].astype(np.float32)
            stats[c] = (np.percentile(x, p_low), np.percentile(x, p_high))
    elif arr.ndim == 3:
        for c in ch_idx:
            x = arr[c].astype(np.float32)
            stats[c] = (np.percentile(x, p_low), np.percentile(x, p_high))
    else:
        raise ValueError(f"Unexpected shape: {arr.shape}")
    return stats


def normalize_with_lo_hi(x, lo, hi, gamma=1.0, eps=1e-8):
    x = x.astype(np.float32)
    y = (x - lo) / (hi - lo + eps)
    y = np.clip(y, 0.0, 1.0)
    if gamma != 1.0:
        y = y ** (1.0 / gamma)
    return y


def compose_rgb_ch0123(thg, nadh, shg, fad, shg_scale=1.0):
    """
    你的定义：
      R = THG + FAD
      G = FAD + SHG*scale
      B = THG + NADH
    """
    R = thg + fad
    G = fad + (shg * shg_scale)
    B = thg + nadh
    rgb = np.stack([R, G, B], axis=-1)
    return np.clip(rgb, 0.0, 1.0)


def contrast_stretch_to_uint8(rgb: np.ndarray, p_low=1.0, p_high=99.0, gamma=None) -> np.ndarray:
    """
    百分位对比度拉伸到 8-bit（每通道单独拉伸）
    """
    x = rgb.astype(np.float32)
    out = np.empty_like(x, dtype=np.float32)
    for c in range(3):
        lo = np.percentile(x[..., c], p_low)
        hi = np.percentile(x[..., c], p_high)
        if hi <= lo + 1e-6:
            out[..., c] = np.clip(x[..., c], 0, 255)
            continue
        y = (x[..., c] - lo) / (hi - lo)
        y = np.clip(y, 0.0, 1.0)
        if gamma is not None:
            y = np.power(y, gamma)
        out[..., c] = y * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)


def raw_tif_to_rgb_uint8_stack(tif_path: Path) -> np.ndarray:
    """
    读 raw (T,C,H,W) 或 (C,H,W)，输出 (T,H,W,3) uint8
    """
    arr = tiff.imread(tif_path)
    if arr.ndim == 3:
        arr = arr[None, ...]  # (1,C,H,W)
    if arr.ndim != 4:
        raise ValueError(f"Expected (T,C,H,W), got {arr.shape} for {tif_path}")

    T, C, H, W = arr.shape
    if C < 4:
        raise ValueError(f"Need >=4 channels, got C={C} for {tif_path.name}")

    stats = compute_global_lo_hi_selected(arr, KEEP_CH, RAW_P_LOW, RAW_P_HIGH)

    print(f"[RGB] {tif_path.name} shape={arr.shape} dtype={arr.dtype}")
    for c in KEEP_CH:
        lo, hi = stats[c]
        print(f"  ch{c} lo/hi(p{RAW_P_LOW},p{RAW_P_HIGH}) = {lo:.3f}/{hi:.3f}")

    lo0, hi0 = stats[0]
    lo1, hi1 = stats[1]
    lo2, hi2 = stats[2]
    lo3, hi3 = stats[3]

    out = np.zeros((T, H, W, 3), dtype=np.uint8)

    for t in range(T):
        frame = arr[t]  # (C,H,W)

        thg  = normalize_with_lo_hi(frame[0], lo0, hi0, gamma=RAW_GAMMA)
        nadh = normalize_with_lo_hi(frame[1], lo1, hi1, gamma=RAW_GAMMA)
        shg  = normalize_with_lo_hi(frame[2], lo2, hi2, gamma=RAW_GAMMA)
        fad  = normalize_with_lo_hi(frame[3], lo3, hi3, gamma=RAW_GAMMA)

        thg_corr = thg.copy()
        mask_green = shg > SHG_PRESENT_THR
        if mask_green.any():
            thg_corr[mask_green] = thg_corr[mask_green] - (THG_MINUS_SHG_SCALE * shg[mask_green])
        thg_corr = np.clip(thg_corr, 0.0, 1.0)

        rgb = compose_rgb_ch0123(thg_corr, nadh, shg, fad, shg_scale=SHG_SCALE)
        out[t] = (rgb * 255.0 + 0.5).astype(np.uint8)

    print(f"[OK] built RGB stack in-memory | size={W}x{H} frames={T}")
    print(f"     params: SHG_SCALE={SHG_SCALE}, SHG_PRESENT_THR={SHG_PRESENT_THR}, THG_MINUS_SHG_SCALE={THG_MINUS_SHG_SCALE}")
    return out


def save_rgb_stack_as_frames(rgb_stack_uint8: np.ndarray, out_dir: Path):
    """
    输入: (T,H,W,3) uint8
    输出: out_dir/frame_000000.png ...
    """
    if rgb_stack_uint8.ndim != 4 or rgb_stack_uint8.shape[-1] != 3:
        raise ValueError(f"Expected (T,H,W,3), got {rgb_stack_uint8.shape}")

    out_dir.mkdir(parents=True, exist_ok=True)
    T = rgb_stack_uint8.shape[0]

    for i in range(T):
        rgb = rgb_stack_uint8[i]  # (H,W,3), uint8

        if ENABLE_FRAME_CONTRAST_STRETCH:
            rgb8 = contrast_stretch_to_uint8(rgb, p_low=FRAME_P_LOW, p_high=FRAME_P_HIGH, gamma=FRAME_GAMMA)
        else:
            rgb8 = rgb  # 已经是 8-bit

        out_path = out_dir / f"frame_{i:0{EXPORT_PAD}d}.{EXPORT_FORMAT}"

        # 用 tifffile 写 png/tif 都可以
        tiff.imwrite(str(out_path), rgb8)

    print(f"[Done] Saved {T} frames to: {out_dir}")
    print(f"       dtype={rgb_stack_uint8.dtype}, shape={rgb_stack_uint8.shape}")
    if ENABLE_FRAME_CONTRAST_STRETCH:
        print(f"       frame stretch: p_low={FRAME_P_LOW}, p_high={FRAME_P_HIGH}, gamma={FRAME_GAMMA}")
    else:
        print("       frame stretch: DISABLED")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for p in RAW_TIFS:
        tif_path = Path(p)
        if not tif_path.exists():
            print(f"[Skip] not found: {tif_path}")
            continue

        # 1) raw -> rgb stack (uint8) in memory
        rgb_stack = raw_tif_to_rgb_uint8_stack(tif_path)

        # 2) 直接切片输出到文件夹
        out_frames_dir = OUT_DIR / f"{tif_path.stem}_rgb_frames"
        save_rgb_stack_as_frames(rgb_stack, out_frames_dir)


if __name__ == "__main__":
    main()
