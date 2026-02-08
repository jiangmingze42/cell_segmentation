#!/usr/bin/env python3
"""
infer_folder_overlay.py

Main model (SAM3 or SAM3+LoRA) inference on images in a folder (inference only),
then overlay results onto frames from an input TIF (base image kept),
and save both originals + overlays.

Auto-handle TIF formats:
- (T,3,H,W)   OR (T,H,W,3) OR (3,H,W) OR (H,W,3)
Program converts to unified: (T,3,H,W).

IDs on overlay match id_scores mapping exactly.
Prints per-frame id:score mapping.

Base filters (applied to MAIN model outputs):
1) drop if full mask area < 200 pixels
2) drop if (mask area INSIDE its box) / (box area) < 0.5
3) if two boxes overlap: intersection >= 50% of either box => drop lower-score

Run:
python infer_folder_overlay.py --in_tif ... --img_dir ... --out_dir ... --prompt cell --ckpt ... --bpe ... \
  --lora_config ... --lora_weight ...
"""

import json
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import cv2
import tifffile as tiff

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# LoRA (optional)
import yaml
from lora_layers import LoRAConfig, apply_lora_to_model, load_lora_weights


# --------------------------
# TIF utilities (AUTO FORMAT)
# --------------------------
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
    if frame_cyx.ndim != 3 or frame_cyx.shape[0] != 3:
        raise ValueError(f"Expect (3,H,W), got {frame_cyx.shape}")
    rgb = np.transpose(frame_cyx, (1, 2, 0))
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return rgb


# --------------------------
# Resize output helpers
# --------------------------
def resize_outputs_to_target(masks, boxes, scores, src_size, tgt_size):
    """Resize outputs from src_size (W,H) to tgt_size (W,H)."""
    src_w, src_h = src_size
    tgt_w, tgt_h = tgt_size
    sx = tgt_w / float(src_w)
    sy = tgt_h / float(src_h)

    if boxes is not None and boxes.numel() > 0:
        boxes = boxes.clone()
        boxes[:, 0] *= sx
        boxes[:, 2] *= sx
        boxes[:, 1] *= sy
        boxes[:, 3] *= sy

    if masks is not None and masks.numel() > 0:
        if masks.dtype == torch.bool:
            masks_f = masks.float()
            masks_rs = F.interpolate(masks_f, size=(tgt_h, tgt_w), mode="nearest")
            masks = masks_rs > 0.5
        else:
            masks = F.interpolate(masks, size=(tgt_h, tgt_w), mode="nearest")

    return masks, boxes, scores


# --------------------------
# Filtering helpers
# --------------------------
def suppress_by_box_overlap_ratio(boxes: torch.Tensor, scores: torch.Tensor, overlap_ratio_thr=0.50):
    """Drop lower-score if intersection >= 50% of either box (greedy keep high-score)."""
    if boxes is None or boxes.numel() == 0:
        return []

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = ((x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)).float()

    order = scores.argsort(descending=True)
    keep = []

    for idx in order.tolist():
        if areas[idx].item() <= 0:
            continue

        ok = True
        for j in keep:
            ix1 = max(x1[idx].item(), x1[j].item())
            iy1 = max(y1[idx].item(), y1[j].item())
            ix2 = min(x2[idx].item(), x2[j].item())
            iy2 = min(y2[idx].item(), y2[j].item())

            iw = max(0.0, ix2 - ix1)
            ih = max(0.0, iy2 - iy1)
            inter = iw * ih
            if inter <= 0:
                continue

            aj = areas[j].item()
            ai = areas[idx].item()

            if (inter / (ai + 1e-6) >= overlap_ratio_thr) or (inter / (aj + 1e-6) >= overlap_ratio_thr):
                ok = False
                break

        if ok:
            keep.append(idx)

    return keep

def final_filter_drop_large_instances(label_u16: np.ndarray, id_scores: list, max_area_ratio: float):
    """
    After all masks are selected and merged into label_u16,
    drop instances whose area > H*W*max_area_ratio.
    Then relabel ids to be contiguous starting from 1, and update id_scores accordingly.
    """
    H, W = label_u16.shape
    img_area = float(H * W)
    max_area = img_area * float(max_area_ratio)

    if label_u16.max() == 0:
        return label_u16, id_scores, 0

    # compute area per id
    labels = [int(x) for x in np.unique(label_u16) if x != 0]
    areas = {lab: float((label_u16 == lab).sum()) for lab in labels}

    # decide keep
    keep_old = [lab for lab in labels if areas[lab] <= max_area]
    dropped = len(labels) - len(keep_old)
    if dropped == 0:
        return label_u16, id_scores, 0

    # build mapping old_id -> new_id (keep ascending old id to keep behavior stable)
    keep_old_sorted = sorted(keep_old)
    old2new = {old: (i + 1) for i, old in enumerate(keep_old_sorted)}

    # relabel mask
    new_label = np.zeros_like(label_u16, dtype=np.uint16)
    for old, new in old2new.items():
        new_label[label_u16 == old] = np.uint16(new)

    # update id_scores (keep original score for kept ids)
    # id_scores is list of {"id": old_id, "score": ...}
    new_id_scores = []
    for d in id_scores:
        old_id = int(d["id"])
        if old_id in old2new:
            new_id_scores.append({"id": int(old2new[old_id]), "score": float(d["score"])})

    return new_label, new_id_scores, dropped


def compute_mask_area_inside_boxes(mb: torch.Tensor, boxes: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """Return mask pixel count INSIDE each box. mb: (N,H,W) bool."""
    N = mb.shape[0]
    areas_in = torch.zeros((N,), device=mb.device, dtype=torch.float32)

    b = boxes.clone()
    b[:, 0] = b[:, 0].clamp(0, W)
    b[:, 2] = b[:, 2].clamp(0, W)
    b[:, 1] = b[:, 1].clamp(0, H)
    b[:, 3] = b[:, 3].clamp(0, H)

    xi1 = b[:, 0].floor().long()
    yi1 = b[:, 1].floor().long()
    xi2 = b[:, 2].ceil().long()
    yi2 = b[:, 3].ceil().long()

    for i in range(N):
        x1, y1, x2, y2 = xi1[i].item(), yi1[i].item(), xi2[i].item(), yi2[i].item()
        if x2 <= x1 or y2 <= y1:
            continue
        areas_in[i] = mb[i, y1:y2, x1:x2].sum().float()

    return areas_in


def build_label_mask_from_instances_with_id_scores(
    masks, boxes, scores,
    H, W,
    score_thr=0.0,
    min_mask_area_px=200,
    mask_box_area_ratio_thr=0.50,
    box_overlap_ratio_thr=0.50,
    max_single_mask_area_ratio=(1.0 / 20.0),   # <<< NEW
):
    """
    Build initial label mask from MAIN model outputs.
    Returns:
      label_u16: (H,W) uint16, ids 1..K
      id_scores: list[dict], each {"id": int, "score": float} where id matches overlay id
    """
    if masks is None or masks.numel() == 0:
        return np.zeros((H, W), np.uint16), []

    sel = scores >= float(score_thr)
    masks, boxes, scores = masks[sel], boxes[sel], scores[sel]
    if masks.shape[0] == 0:
        return np.zeros((H, W), np.uint16), []

    mb = masks[:, 0]
    if mb.dtype != torch.bool:
        mb = mb > 0.0

    # rule 1: min full mask area
    mask_area_full = mb.flatten(1).sum(dim=1).float()
    sel = mask_area_full >= float(min_mask_area_px)
    boxes, scores, mb = boxes[sel], scores[sel], mb[sel]
    if mb.shape[0] == 0:
        return np.zeros((H, W), np.uint16), []

    # rule 1.5: drop if single mask too large
    img_area = float(H * W)
    max_area = img_area * float(max_single_mask_area_ratio)
    mask_area_full = mb.flatten(1).sum(dim=1).float()
    sel = mask_area_full <= max_area
    boxes, scores, mb = boxes[sel], scores[sel], mb[sel]
    if mb.shape[0] == 0:
        return np.zeros((H, W), np.uint16), []

    # rule 2: (mask inside box) / (box area)
    x1, y1, x2, y2 = boxes.T
    box_areas = ((x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)).float()
    mask_in_box = compute_mask_area_inside_boxes(mb, boxes, H=H, W=W)

    ratio = torch.zeros_like(mask_in_box)
    valid_box = box_areas > 0
    ratio[valid_box] = mask_in_box[valid_box] / (box_areas[valid_box] + 1e-6)

    sel = ratio >= float(mask_box_area_ratio_thr)
    boxes, scores, mb = boxes[sel], scores[sel], mb[sel]
    if mb.shape[0] == 0:
        return np.zeros((H, W), np.uint16), []

    # rule 3: box overlap suppression
    keep_idx = suppress_by_box_overlap_ratio(boxes, scores, overlap_ratio_thr=box_overlap_ratio_thr)
    if len(keep_idx) == 0:
        return np.zeros((H, W), np.uint16), []

    boxes, scores, mb = boxes[keep_idx], scores[keep_idx], mb[keep_idx]

    # label fill: high score first, do not overwrite
    order = scores.argsort(descending=True).tolist()
    label = torch.zeros((H, W), device=mb.device, dtype=torch.int32)

    id_scores = []
    cur_id = 0
    for i in order:
        m = mb[i]
        if not bool(m.any()):
            continue
        empty = label == 0
        fill = m & empty
        if not bool(fill.any()):
            continue
        cur_id += 1
        label[fill] = cur_id
        id_scores.append({"id": int(cur_id), "score": float(scores[i].detach().cpu().item())})
        if cur_id >= 65535:
            break

    return label.detach().cpu().numpy().astype(np.uint16), id_scores


# --------------------------
# NEW: add extra SAM3-only masks
# --------------------------
def add_sam3_only_masks(
    label_u16: np.ndarray,
    id_scores: list,
    sam3_masks, sam3_scores,
    overlap_thr: float,
    max_area_ratio: float,
    min_area_px: int,
):
    """
    label_u16: existing label mask (H,W) uint16
    id_scores: existing list of {"id","score"}
    sam3_masks: (N,1,H,W) bool/float
    sam3_scores:(N,) tensor/array
    overlap_thr: require (intersection with existing) / candidate_area < overlap_thr
    max_area_ratio: require candidate_area <= H*W*max_area_ratio  (here = 1/12)
    """
    H, W = label_u16.shape
    img_area = H * W
    max_area = float(img_area) * float(max_area_ratio)

    if sam3_masks is None or sam3_masks.numel() == 0:
        return label_u16, id_scores, 0

    if isinstance(sam3_scores, torch.Tensor):
        scores_t = sam3_scores
    else:
        scores_t = torch.as_tensor(sam3_scores)

    mb = sam3_masks[:, 0]
    if mb.dtype != torch.bool:
        mb = mb > 0.0

    # sort candidates by score desc (nice behavior)
    order = scores_t.argsort(descending=True).tolist()

    existing = (label_u16 > 0)
    next_id = int(label_u16.max()) + 1
    added = 0

    for i in order:
        m = mb[i].detach().cpu().numpy().astype(bool)
        area = float(m.sum())
        if area < float(min_area_px):
            continue
        if area > max_area:
            continue

        inter = float(np.logical_and(m, existing).sum())
        ov = inter / (area + 1e-6)
        if ov >= float(overlap_thr):
            continue

        # add as a new instance; do not overwrite existing pixels
        fill = np.logical_and(m, ~existing)
        if not fill.any():
            continue

        label_u16[fill] = np.uint16(next_id)
        id_scores.append({"id": int(next_id), "score": float(scores_t[i].detach().cpu().item())})
        existing[fill] = True
        next_id += 1
        added += 1

        if next_id >= 65535:
            break

    return label_u16, id_scores, added


# --------------------------
# Overlay (Cellpose-style)
# --------------------------
def overlay_on_rgb(
    rgb_u8: np.ndarray,
    mask_u16: np.ndarray,
    alpha: float = 0.45,
    draw_boundary: bool = True,
    draw_ids: bool = True,
    font_scale: float = 0.45,
):
    H, W, _ = rgb_u8.shape
    base = rgb_u8.astype(np.float32) / 255.0
    out = base.copy()

    labels = [lab for lab in np.unique(mask_u16) if lab != 0]

    for i, lab in enumerate(labels):
        m = (mask_u16 == lab)
        if not m.any():
            continue

        hue = (i * 37) % 180
        color_bgr = cv2.cvtColor(np.uint8([[[hue, 200, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
        color_rgb = color_bgr[::-1].astype(np.float32) / 255.0

        out[m] = out[m] * (1.0 - alpha) + color_rgb[None, :] * alpha

        if draw_boundary:
            mu8 = (m.astype(np.uint8) * 255)
            er = cv2.erode(mu8, np.ones((3, 3), np.uint8), iterations=1)
            bnd = (mu8 > 0) & (er == 0)
            out[bnd] = 1.0

    out_u8 = np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8)

    if draw_ids and len(labels) > 0:
        bgr = cv2.cvtColor(out_u8, cv2.COLOR_RGB2BGR)
        for lab in labels:
            m = (mask_u16 == lab)
            if not m.any():
                continue
            ys, xs = np.where(m)
            cy = int(np.clip(np.round(float(np.mean(ys))), 0, H - 1))
            cx = int(np.clip(np.round(float(np.mean(xs))), 0, W - 1))
            txt = str(int(lab))
            cv2.putText(bgr, txt, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(bgr, txt, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
        out_u8 = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    return out_u8


def format_id_scores(id_scores, max_show=80):
    if not id_scores:
        return "[]"
    show = id_scores[:max_show]
    s = ", ".join([f'{d["id"]}:{d["score"]:.4f}' for d in show])
    if len(id_scores) > max_show:
        s += f", ... (+{len(id_scores)-max_show})"
    return "[" + s + "]"


# --------------------------
# LoRA helper
# --------------------------
def maybe_apply_lora(model, lora_config_path, lora_weights_path):
    if not lora_config_path or not lora_weights_path:
        return model

    with open(lora_config_path, "r") as f:
        cfg = yaml.safe_load(f)

    lcfg = cfg["lora"]
    lora_config = LoRAConfig(
        rank=lcfg["rank"],
        alpha=lcfg["alpha"],
        dropout=0.0,
        target_modules=lcfg["target_modules"],
        apply_to_vision_encoder=lcfg["apply_to_vision_encoder"],
        apply_to_text_encoder=lcfg["apply_to_text_encoder"],
        apply_to_geometry_encoder=lcfg["apply_to_geometry_encoder"],
        apply_to_detr_encoder=lcfg["apply_to_detr_encoder"],
        apply_to_detr_decoder=lcfg["apply_to_detr_decoder"],
        apply_to_mask_decoder=lcfg["apply_to_mask_decoder"],
    )

    model = apply_lora_to_model(model, lora_config)
    load_lora_weights(model, lora_weights_path)
    return model


# --------------------------
# Main
# --------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_tif", type=str, required=True, help="overlay base tif, shape (T,3,H,W) or (T,H,W,3)")
    parser.add_argument("--img_dir", type=str, required=True, help="folder with inference images (any size)")
    parser.add_argument("--out_dir", type=str, required=True)

    parser.add_argument("--prompt", type=str, default="cell")

    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--bpe", type=str, required=True)

    parser.add_argument("--lora_config", type=str, default=None)
    parser.add_argument("--lora_weight", type=str, default=None)  # match your CLI

    parser.add_argument("--score_thr", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=0)

    # overlay style
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument("--draw_boundary", action="store_true", default=True)
    parser.add_argument("--no_draw_boundary", action="store_false", dest="draw_boundary")
    parser.add_argument("--draw_ids", action="store_true", default=True)
    parser.add_argument("--no_draw_ids", action="store_false", dest="draw_ids")
    parser.add_argument("--font_scale", type=float, default=0.45)

    # base filters (MAIN model)
    parser.add_argument("--min_mask_area", type=int, default=200)
    parser.add_argument("--mask_inbox_ratio", type=float, default=0.40)
    parser.add_argument("--box_overlap_ratio", type=float, default=0.50)

    parser.add_argument("--sam3_add", action="store_true", default=True, help="enable adding sam3-only masks")
    parser.add_argument("--no_sam3_add", action="store_false", dest="sam3_add")
    parser.add_argument("--sam3_thr", type=float, default=0.50, help="SAM3-only confidence_threshold")
    parser.add_argument("--sam3_overlap_thr", type=float, default=0.30, help="(intersect with existing) / candidate_area < thr")
    parser.add_argument("--sam3_max_area_ratio", type=float, default=(1.0 / 12.0), help="candidate_area <= img_area * ratio")

    # printing
    parser.add_argument("--print_scores", action="store_true", default=True)
    parser.add_argument("--no_print_scores", action="store_false", dest="print_scores")

    args = parser.parse_args()

    in_tif = Path(args.in_tif)
    img_dir = Path(args.img_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    orig_dir = out_dir / "originals_png"
    ov_dir = out_dir / "overlays_png"
    orig_dir.mkdir(parents=True, exist_ok=True)
    ov_dir.mkdir(parents=True, exist_ok=True)

    # load tif base, auto convert to (T,3,H,W)
    raw = tiff.imread(str(in_tif))
    arr = tif_to_TCYX(raw)
    T, C, H, W = arr.shape
    if C != 3:
        raise ValueError(f"After conversion, expect C=3, got C={C}")

    # gather inference images
    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
    images = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in exts])
    if len(images) == 0:
        raise ValueError(f"No images found in {img_dir}")
    if args.limit > 0:
        images = images[: args.limit]

    # match frames by index
    N = min(len(images), T)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # MAIN model (maybe LoRA)
    model_main = build_sam3_image_model(
        checkpoint_path=args.ckpt,
        bpe_path=args.bpe,
        device=device,
        eval_mode=True,
        load_from_HF=False,
    )
    model_main = maybe_apply_lora(model_main, args.lora_config, args.lora_weight)
    model_main.to(device)
    processor_main = Sam3Processor(model_main, confidence_threshold=0.4)

    # SAM3-only model (pure SAM3, no LoRA) for adding masks
    processor_sam3 = None
    if args.sam3_add:
        model_sam3 = build_sam3_image_model(
            checkpoint_path=args.ckpt,
            bpe_path=args.bpe,
            device=device,
            eval_mode=True,
            load_from_HF=False,
        )
        model_sam3.to(device)
        processor_sam3 = Sam3Processor(model_sam3, confidence_threshold=float(args.sam3_thr))

    scores_jsonl = out_dir / "scores.jsonl"

    INFER_SIZE = (1008, 1008)   # (W,H) inference size
    OUT_SIZE = (W, H)           # overlay coords match base tif size

    with open(scores_jsonl, "w", encoding="utf-8") as fout:
        for t in range(N):
            base_rgb = to_rgb_u8_from_CYX(arr[t])
            Image.fromarray(base_rgb, mode="RGB").save(orig_dir / f"t{t:03d}.png")

            infer_img = Image.open(images[t]).convert("RGB")
            infer_img_1008 = infer_img.resize(INFER_SIZE, Image.BILINEAR)

            # ---- MAIN inference ----
            state = processor_main.set_image(infer_img_1008)
            out = processor_main.set_text_prompt(state=state, prompt=args.prompt)
            masks, boxes, scores = out["masks"], out["boxes"], out["scores"]

            masks, boxes, scores = resize_outputs_to_target(
                masks, boxes, scores,
                src_size=INFER_SIZE,
                tgt_size=(OUT_SIZE[0], OUT_SIZE[1]),
            )

            label_u16, id_scores = build_label_mask_from_instances_with_id_scores(
                masks, boxes, scores,
                H=H, W=W,
                score_thr=args.score_thr,
                min_mask_area_px=args.min_mask_area,
                mask_box_area_ratio_thr=args.mask_inbox_ratio,
                box_overlap_ratio_thr=args.box_overlap_ratio,
                max_single_mask_area_ratio=float(1/20),
            )


            # ---- SAM3-only add masks (your rule) ----
            added_n = 0
            if processor_sam3 is not None:
                state2 = processor_sam3.set_image(infer_img_1008)
                out2 = processor_sam3.set_text_prompt(state=state2, prompt=args.prompt)
                masks2, boxes2, scores2 = out2["masks"], out2["boxes"], out2["scores"]

                # resize masks2 back to base size
                masks2, boxes2, scores2 = resize_outputs_to_target(
                    masks2, boxes2, scores2,
                    src_size=INFER_SIZE,
                    tgt_size=(OUT_SIZE[0], OUT_SIZE[1]),
                )

                label_u16, id_scores, added_n = add_sam3_only_masks(
                    label_u16=label_u16,
                    id_scores=id_scores,
                    sam3_masks=masks2,
                    sam3_scores=scores2,
                    overlap_thr=float(args.sam3_overlap_thr),
                    max_area_ratio=float(args.sam3_max_area_ratio),  # default 1/12
                    min_area_px=int(args.min_mask_area),
                )
                # ---- FINAL: drop masks larger than 1/20 after all selections ----
                label_u16, id_scores, dropped_big = final_filter_drop_large_instances(
                    label_u16=label_u16,
                    id_scores=id_scores,
                    max_area_ratio=float(1.0 / 20.0),
                )

            # ---- overlay ----
            ov_u8 = overlay_on_rgb(
                base_rgb,
                label_u16,
                alpha=args.alpha,
                draw_boundary=args.draw_boundary,
                draw_ids=args.draw_ids,
                font_scale=args.font_scale,
            )
            Image.fromarray(ov_u8, mode="RGB").save(ov_dir / f"t{t:03d}_overlay.png")

            # if args.print_scores:
            #     print(
            #         f"[ID:SCORE] t={t:03d} kept_n={int(label_u16.max())} "
            #         f"(added_sam3={added_n}) {format_id_scores(id_scores)}",
            #         flush=True,
            #     )

            # fout.write(json.dumps({
            #     "t": t,
            #     "base_tif": in_tif.name,
            #     "base_shape_raw": tuple(raw.shape),
            #     "base_shape_TCYX": tuple(arr.shape),
            #     "infer_image": images[t].name,
            #     "raw_n_main": int(masks.shape[0]) if masks is not None else 0,
            #     "kept_n_total": int(label_u16.max()),
            #     "added_from_sam3": int(added_n),
            #     "id_scores": id_scores,
            # }) + "\n")

            if (t + 1) % 10 == 0 or t == N - 1:
                print(f"[INFO] t={t+1}/{N} kept_instances={int(label_u16.max())} added_sam3={added_n}", flush=True)

    print("[OK] originals ->", orig_dir)
    print("[OK] overlays   ->", ov_dir)
    print("[OK] scores     ->", scores_jsonl)


if __name__ == "__main__":
    main()
