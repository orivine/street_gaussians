#!/usr/bin/env python3
"""
Export PSNR* masks as PNG files for manual inspection.

This script reuses the existing PSNR* definition in lib/utils/waymo_psnr_star.py
and writes masks under:

    <cfg.model_path>/<output_subdir>/<split>/

Each split contains:
    binary/<image_name>.png   - 8-bit binary mask
    overlay/<image_name>.png  - mask blended on top of the input image
    summary.json              - export metadata and per-view mask stats

Repo config arguments such as --config and YACS opts are passed through to the
standard config loader.
"""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

def parse_script_args():
    parser = argparse.ArgumentParser(
        description=(
            "Export PSNR* masks to PNG. Repo config args such as --config and "
            "YACS opts are passed through after the script-specific arguments."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--split",
        choices=("train", "test", "both"),
        default="test",
        help="Which dataset split to export.",
    )
    parser.add_argument(
        "--output-subdir",
        default="psnr_star_mask",
        help="Subdirectory under cfg.model_path used for exported masks.",
    )
    parser.add_argument(
        "--no-overlay",
        action="store_true",
        help="Only export binary masks and skip RGB overlay images.",
    )
    parser.add_argument(
        "--overlay-alpha",
        type=float,
        default=0.45,
        help="Blend weight used for mask overlays.",
    )
    parser.add_argument(
        "--expand-factor",
        type=float,
        default=1.5,
        help="Bounding box length/width expansion factor for PSNR* masks.",
    )
    parser.add_argument(
        "--speed-threshold",
        type=float,
        default=1.0,
        help="Minimum object speed in m/s included in the PSNR* mask.",
    )
    parser.add_argument(
        "--resolution-scale",
        type=float,
        default=1.0,
        help="Resolution scale key used by evaluation. 1.0 matches metrics_complete.py.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=-1,
        help="Optional cap on exported views per split for quick spot checks.",
    )
    args, repo_args = parser.parse_known_args()

    if not 0.0 <= args.overlay_alpha <= 1.0:
        parser.error("--overlay-alpha must be in [0, 1].")
    if args.resolution_scale <= 0:
        parser.error("--resolution-scale must be positive.")
    if args.limit == 0:
        parser.error("--limit must be -1 or a positive integer.")

    # Keep the exporter read-only by default through the repo's YACS-style opts.
    if "mode" not in repo_args:
        repo_args = repo_args + ["mode", "test"]

    # Hand the remaining args to the standard repo config parser.
    sys.argv = [sys.argv[0]] + repo_args
    return args


SCRIPT_ARGS = parse_script_args()

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from lib.config import cfg  # noqa: E402
from lib.datasets.waymo_full_readers import readWaymoFullInfo  # noqa: E402
from lib.utils.waymo_psnr_star import compute_psnr_star_masks  # noqa: E402


MASK_COLOR = np.array([255.0, 64.0, 64.0], dtype=np.float32)


def compute_scaled_resolution(width, height, resolution_scale):
    scale = min(1.0, 1600.0 / float(width))
    scale = scale / float(resolution_scale)
    return int(width * scale), int(height * scale)


def build_mask_views(cam_infos, resolution_scale, limit=-1):
    cam_infos = list(sorted(cam_infos, key=lambda cam: cam.uid))
    if limit > 0:
        cam_infos = cam_infos[:limit]

    views = []
    for cam_info in cam_infos:
        width, height = compute_scaled_resolution(
            width=cam_info.width,
            height=cam_info.height,
            resolution_scale=resolution_scale,
        )
        views.append(
            SimpleNamespace(
                image_name=cam_info.image_name,
                image_width=width,
                image_height=height,
            )
        )

    return cam_infos, views


def save_mask_png(mask, path):
    Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(path)


def save_overlay_png(cam_info, mask, path, alpha):
    width = mask.shape[1]
    height = mask.shape[0]
    image = cam_info.image.convert("RGB").resize((width, height), resample=Image.BILINEAR)
    image_np = np.asarray(image, dtype=np.float32)

    overlay = image_np.copy()
    overlay[mask] = (1.0 - alpha) * overlay[mask] + alpha * MASK_COLOR
    Image.fromarray(np.clip(overlay, 0.0, 255.0).astype(np.uint8), mode="RGB").save(path)


def export_split(split_name, cam_infos):
    split_root = Path(cfg.model_path) / SCRIPT_ARGS.output_subdir / split_name
    binary_dir = split_root / "binary"
    overlay_dir = split_root / "overlay"

    binary_dir.mkdir(parents=True, exist_ok=True)
    if not SCRIPT_ARGS.no_overlay:
        overlay_dir.mkdir(parents=True, exist_ok=True)

    selected_cam_infos, mask_views = build_mask_views(
        cam_infos=cam_infos,
        resolution_scale=SCRIPT_ARGS.resolution_scale,
        limit=SCRIPT_ARGS.limit,
    )
    cam_infos_by_name = {cam_info.image_name: cam_info for cam_info in selected_cam_infos}

    print(f"[{split_name}] preparing {len(mask_views)} views")
    masks = compute_psnr_star_masks(
        source_path=cfg.source_path,
        cam_infos=mask_views,
        use_tracker=cfg.data.get("use_tracker", False),
        expand_factor=SCRIPT_ARGS.expand_factor,
        speed_threshold=SCRIPT_ARGS.speed_threshold,
    )

    summary = {
        "split": split_name,
        "source_path": cfg.source_path,
        "model_path": cfg.model_path,
        "output_dir": str(split_root),
        "expand_factor": SCRIPT_ARGS.expand_factor,
        "speed_threshold": SCRIPT_ARGS.speed_threshold,
        "resolution_scale": SCRIPT_ARGS.resolution_scale,
        "num_views": len(mask_views),
        "non_empty_views": 0,
        "views": {},
    }

    for image_name, mask in masks.items():
        save_mask_png(mask=mask, path=binary_dir / f"{image_name}.png")

        if not SCRIPT_ARGS.no_overlay:
            save_overlay_png(
                cam_info=cam_infos_by_name[image_name],
                mask=mask,
                path=overlay_dir / f"{image_name}.png",
                alpha=SCRIPT_ARGS.overlay_alpha,
            )

        pixel_count = int(mask.sum())
        if pixel_count > 0:
            summary["non_empty_views"] += 1

        summary["views"][image_name] = {
            "height": int(mask.shape[0]),
            "width": int(mask.shape[1]),
            "mask_pixels": pixel_count,
            "mask_ratio": float(mask.mean()),
        }

    with open(split_root / "summary.json", "w") as fp:
        json.dump(summary, fp, indent=2)

    print(
        f"[{split_name}] exported {len(masks)} masks, "
        f"{summary['non_empty_views']} with moving-object pixels"
    )
    print(f"[{split_name}] output: {split_root}")


def main():
    if cfg.data.get("type", "Colmap") != "Waymo":
        raise ValueError("export_psnr_star_masks.py currently supports cfg.data.type == 'Waymo' only.")

    # Keep export read-only: avoid training-mode side effects such as regenerating assets.
    cfg.mode = "test"

    scene_info = readWaymoFullInfo(cfg.source_path, **cfg.data)

    split_to_cams = {
        "train": scene_info.train_cameras,
        "test": scene_info.test_cameras,
    }

    splits = ("train", "test") if SCRIPT_ARGS.split == "both" else (SCRIPT_ARGS.split,)
    for split_name in splits:
        export_split(split_name=split_name, cam_infos=split_to_cams[split_name])


if __name__ == "__main__":
    main()
