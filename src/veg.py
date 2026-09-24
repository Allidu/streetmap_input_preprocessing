#!/usr/bin/env python3
"""Area-mode FIRST filter: Mapillary photos -> building/vegetation masks.

Reads a get_mapillary.py area-mode JSON object and preserves its structure, only
replacing its ``mapillary`` array with visually accepted original image records.
Images are loaded from a record's valid local_path/image_path if present, or
from its existing thumb_original_url; no separate image-input folder is needed.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
from pathlib import Path

import cv2
import numpy as np
import requests
from PIL import Image, ImageOps

MODEL_ID = "facebook/mask2former-swin-large-cityscapes-panoptic"
MAX_IMAGE_BYTES = 30 * 1024 * 1024
THUMB_W = 480


def mask_from_segments(segments, category: str, shape: tuple[int, int]) -> np.ndarray:
    """Union all predicted segments with an exact semantic category label."""
    h, w = shape
    merged = np.zeros((h, w), dtype=bool)
    for segment in segments:
        if str(segment.get("label", "")).lower() != category.lower():
            continue
        mask = np.asarray(segment["mask"])
        if mask.ndim == 3:
            mask = mask[..., 0]
        if mask.shape != (h, w):
            mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        merged |= mask > 0
    return merged


def calculate_metrics(building: np.ndarray, vegetation: np.ndarray) -> dict:
    h, w = building.shape
    # Center-region vegetation coverage is a *proxy*, NOT true tree-on-building occlusion.
    y0, y1 = int(0.15 * h), max(int(0.85 * h), int(0.15 * h) + 1)
    x0, x1 = int(0.15 * w), max(int(0.85 * w), int(0.15 * w) + 1)
    return {
        "building_frac": float(building.mean()),
        "vegetation_frac": float(vegetation.mean()),
        "center_vegetation_frac": float(vegetation[y0:y1, x0:x1].mean()),
    }


def assess(metrics: dict, min_building_frac: float, max_building_frac: float,
           max_center_vegetation_frac: float) -> str | None:
    if metrics["building_frac"] < min_building_frac:
        return "low_building_coverage"
    if metrics["building_frac"] > max_building_frac:
        return "excessive_building_coverage_review"
    if metrics["center_vegetation_frac"] > max_center_vegetation_frac:
        return "vegetation_dominates_center"
    return None


def open_image(img: dict, session: requests.Session) -> Image.Image:
    # Accept future upstream workflows that provide local file paths, while
    # defaulting to get_mapillary.py's URL-based records as they exist today.
    for key in ("local_path", "image_path"):
        candidate = img.get(key)
        if candidate and Path(candidate).is_file():
            with Image.open(candidate) as raw:
                return ImageOps.exif_transpose(raw).convert("RGB")
    url = img.get("thumb_original_url")
    if not isinstance(url, str) or not url.startswith(("https://", "http://")):
        raise ValueError("missing_image_url_or_local_file")
    with session.get(url, timeout=(10, 60), stream=True) as response:
        response.raise_for_status()
        chunks = []
        total = 0
        for chunk in response.iter_content(1024 * 1024):
            total += len(chunk)
            if total > MAX_IMAGE_BYTES:
                raise ValueError("image_exceeds_30mb_limit")
            chunks.append(chunk)
    with Image.open(io.BytesIO(b"".join(chunks))) as raw:
        return ImageOps.exif_transpose(raw).convert("RGB")


def resize_for_model(image: Image.Image, max_edge: int) -> Image.Image:
    if max_edge <= 0:
        raise ValueError("max_edge must be positive")
    if max(image.size) <= max_edge:
        return image
    image = image.copy()
    image.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
    return image


def visualize(image: Image.Image, building: np.ndarray, vegetation: np.ndarray,
              image_id: str, metrics: dict, reason: str | None) -> np.ndarray:
    bgr = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]
    new_h = max(1, int(round(h * THUMB_W / w)))
    bgr = cv2.resize(bgr, (THUMB_W, new_h), interpolation=cv2.INTER_AREA)
    b = cv2.resize(building.astype(np.uint8), (THUMB_W, new_h), interpolation=cv2.INTER_NEAREST) > 0
    v = cv2.resize(vegetation.astype(np.uint8), (THUMB_W, new_h), interpolation=cv2.INTER_NEAREST) > 0
    green = np.array([0, 255, 0], dtype=np.float32)
    orange = np.array([0, 140, 255], dtype=np.float32)
    bgr[b] = (0.65 * bgr[b] + 0.35 * green).astype(np.uint8)
    bgr[v] = (0.65 * bgr[v] + 0.35 * orange).astype(np.uint8)
    cv2.rectangle(bgr, (0, 0), (THUMB_W, 75), (0, 0, 0), -1)
    lines = (f"{'REJECT' if reason else 'ACCEPT'}: {reason or '-'}",
             f"ID {image_id}",
             f"bldg={metrics['building_frac']:.2f} veg={metrics['vegetation_frac']:.2f} ctr={metrics['center_vegetation_frac']:.2f}")
    for j, line in enumerate(lines):
        cv2.putText(bgr, line[:65], (6, 19 + 23 * j), cv2.FONT_HERSHEY_SIMPLEX,
                    0.50, (255, 255, 255), 1, cv2.LINE_AA)
    return bgr


def save_sheet(thumbs: list[np.ndarray], path: Path, columns: int = 4) -> None:
    if not thumbs:
        return
    pad = 8
    cell_h = max(t.shape[0] for t in thumbs)
    rows = math.ceil(len(thumbs) / columns)
    sheet = np.full((rows * (cell_h + pad) + pad, columns * (THUMB_W + pad) + pad, 3), 255, np.uint8)
    for i, thumb in enumerate(thumbs):
        x = pad + (i % columns) * (THUMB_W + pad)
        y = pad + (i // columns) * (cell_h + pad)
        sheet[y:y + thumb.shape[0], x:x + thumb.shape[1]] = thumb
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), sheet):
        raise OSError(f"Could not write contact sheet: {path}")


def filter_area(data: dict, segmenter, session: requests.Session, *,
                min_building_frac: float, max_building_frac: float,
                max_center_vegetation_frac: float, max_edge: int,
                max_sheet_images: int = 80):
    if data.get("mode") != "area" or not isinstance(data.get("mapillary"), list):
        raise ValueError("Expected area-mode get_mapillary JSON with a mapillary image list")
    accepted, rejected, rows, sheet_items = [], [], [], []
    evaluated_count = 0
    for img in data["mapillary"]:
        image_id = str(img.get("id", "unknown"))
        try:
            image = resize_for_model(open_image(img, session), max_edge)
            segments = segmenter(image)
            shape = (image.height, image.width)
            building = mask_from_segments(segments, "building", shape)
            vegetation = mask_from_segments(segments, "vegetation", shape)
            metrics = calculate_metrics(building, vegetation)
            reason = assess(metrics, min_building_frac, max_building_frac,
                            max_center_vegetation_frac)
            evaluated_count += 1
            sheet_items.append((visualize(image, building, vegetation, image_id, metrics, reason),
                                reason is None))
        except Exception as exc:
            # Bad/missing photos must not silently bypass a mandatory first filter.
            # These are separately marked as errors so the user can retry them.
            reason = "image_or_segmentation_error"
            metrics = {"error": f"{type(exc).__name__}: {exc}"}
        record = dict(img)  # keep original ID, URL, GPS, heading, capture time, camera parameters
        record["visual_filter"] = {"accepted": reason is None, "reason": reason, **metrics}
        (accepted if reason is None else rejected).append(record)
        rows.append((image_id, int(reason is None), reason or "-",
                     metrics.get("building_frac"), metrics.get("vegetation_frac"),
                     metrics.get("center_vegetation_frac"), metrics.get("error", "")))
        print(f"{image_id}: {'KEEP' if reason is None else 'REJECT'} ({reason or '-'})")
    if data["mapillary"] and evaluated_count == 0:
        raise RuntimeError("No images could be evaluated. Check Mapillary image URLs, connectivity, and model installation.")
    # Copy rather than mutating the fetched raw JSON; preserve all OSM metadata.
    result = dict(data)
    result["mapillary"] = accepted
    result["visual_filter_summary"] = {"input_count": len(data["mapillary"]),
                                       "accepted_count": len(accepted),
                                       "rejected_count": len(rejected),
                                       "evaluated_count": evaluated_count}
    return result, rejected, rows, select_sheet_thumbs(sheet_items, max_sheet_images)


def select_sheet_thumbs(items, max_images: int) -> list:
    """Keep every thumbnail when it fits; otherwise sample accepted and rejected evenly."""
    if max_images <= 0 or len(items) <= max_images:
        return [thumb for thumb, _accepted in items]
    accepted = [thumb for thumb, ok in items if ok]
    rejected = [thumb for thumb, ok in items if not ok]
    n_acc = min(len(accepted), max_images // 2)
    n_rej = min(len(rejected), max_images - n_acc)
    leftover = max_images - (n_acc + n_rej)
    if leftover and len(accepted) > n_acc:
        n_acc = min(len(accepted), n_acc + leftover)
    elif leftover and len(rejected) > n_rej:
        n_rej = min(len(rejected), n_rej + leftover)

    def take(seq, n):
        if n <= 0:
            return []
        if n >= len(seq):
            return list(seq)
        if n == 1:
            return [seq[len(seq) // 2]]
        idxs = [round(i * (len(seq) - 1) / (n - 1)) for i in range(n)]
        return [seq[i] for i in idxs]

    return take(accepted, n_acc) + take(rejected, n_rej)


def main():
    parser = argparse.ArgumentParser(description="First-stage visual filter for Mapillary AREA mode")
    parser.add_argument("--input", required=True, help="Raw area_fetch.json")
    parser.add_argument("--output", required=True, help="Area-mode JSON with accepted images only")
    parser.add_argument("--rejected", required=True, help="Rejected image records and reasons")
    parser.add_argument("--csv", required=True, help="Metrics for every image")
    parser.add_argument("--contact-sheet", required=True, help="Small inspection gallery")
    parser.add_argument("--min-building-frac", type=float, default=0.08)
    parser.add_argument("--max-building-frac", type=float, default=0.93)
    parser.add_argument("--max-center-vegetation-frac", type=float, default=0.55)
    parser.add_argument("--max-edge", type=int, default=1024)
    parser.add_argument("--max-sheet-images", type=int, default=80)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--device", default="-1", help="-1 for CPU, 0 for first CUDA GPU, or mps")
    args = parser.parse_args()
    for name in ("min_building_frac", "max_building_frac", "max_center_vegetation_frac"):
        value = getattr(args, name)
        if not 0 <= value <= 1:
            parser.error(f"--{name.replace('_','-')} must be between 0 and 1")
    if args.min_building_frac >= args.max_building_frac:
        parser.error("Minimum building fraction must be below maximum")
    # Lazy import permits unit tests without downloading a large ML model.
    try:
        from transformers import pipeline
    except ImportError as exc:
        raise RuntimeError("Install transformers, torch, and vision dependencies before running veg.py") from exc
    device = int(args.device) if args.device.lstrip("-").isdigit() else args.device
    segmenter = pipeline("image-segmentation", model=args.model_id, device=device)
    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    with requests.Session() as session:
        result, rejected, rows, thumbs = filter_area(
            data, segmenter, session,
            min_building_frac=args.min_building_frac,
            max_building_frac=args.max_building_frac,
            max_center_vegetation_frac=args.max_center_vegetation_frac,
            max_edge=args.max_edge, max_sheet_images=args.max_sheet_images)
    for filename, content in ((args.output, result), (args.rejected, rejected)):
        out = Path(filename)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(content, indent=2), encoding="utf-8")
    csv_out = Path(args.csv)
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    with csv_out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(("id", "accepted", "reason", "building_frac", "vegetation_frac",
                    "center_vegetation_frac", "error"))
        w.writerows(rows)
    save_sheet(thumbs, Path(args.contact_sheet))
    print(f"Visual stage: {len(result['mapillary'])} accepted, {len(rejected)} rejected")
    print(f"Accepted area JSON: {args.output}")


if __name__ == "__main__":
    main()
