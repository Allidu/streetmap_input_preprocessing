#!/usr/bin/env python3
"""Pick a small set of strong building views from geometrically accepted photos.

Starts from the Thompson Street geometric-accepted list. Ranks by building
visibility, drops near-duplicate Mapillary frames, then caps images per OSM
building. Does not rerun segmentation or geometric scoring.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from html import escape
from pathlib import Path

import yaml

import cv2
import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont

import filter_metadata as geo


ROOT = Path(__file__).resolve().parents[1]
DUP_DISTANCE_M = 8.0
DUP_HEADING_DEG = 20.0
MAX_PER_BUILDING = 5
# Weights sum to 1. Building fraction is the dominant term.
W_BUILDING = 0.50
W_CENTER_VEG = 0.20
W_VEG = 0.10
W_GEOM = 0.15
W_SHARP = 0.05


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def latlon(img: dict) -> tuple[float, float] | None:
    coords = (img.get("computed_geometry") or {}).get("coordinates")
    if not isinstance(coords, list) or len(coords) < 2:
        return None
    try:
        return float(coords[1]), float(coords[0])
    except (TypeError, ValueError):
        return None


def heading_of(img: dict) -> float | None:
    value = img.get("computed_compass_angle")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def frac(value) -> float:
    if isinstance(value, (int, float)) and math.isfinite(value):
        return float(value)
    return 0.0


def download_image(img: dict, session: requests.Session, dest: Path) -> bool:
    if dest.is_file() and dest.stat().st_size > 0:
        return True
    url = img.get("thumb_original_url")
    if not isinstance(url, str) or not url.startswith("http"):
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        response = session.get(url, timeout=(10, 60))
        response.raise_for_status()
        raw = dest.with_suffix(".part")
        raw.write_bytes(response.content)
        with Image.open(raw) as image:
            image = image.convert("RGB")
            image.thumbnail((960, 960), Image.Resampling.LANCZOS)
            image.save(dest, quality=85)
        raw.unlink(missing_ok=True)
        return True
    except Exception as exc:
        print(f"download_failed {img.get('id')} {type(exc).__name__}")
        return False


def sharpness(path: Path) -> float | None:
    if not path.is_file():
        return None
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return None
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def cluster_duplicates(records: list[dict], distance_m: float, heading_deg: float) -> list[int]:
    """Connected components of images within distance_m and heading_deg."""
    parent = list(range(len(records)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    poses = []
    for index, img in enumerate(records):
        pose = latlon(img)
        heading = heading_of(img)
        poses.append((pose, heading))
    for i, (pose_a, heading_a) in enumerate(poses):
        if pose_a is None or heading_a is None:
            continue
        lat_a, lon_a = pose_a
        for j in range(i + 1, len(records)):
            pose_b, heading_b = poses[j]
            if pose_b is None or heading_b is None:
                continue
            if geo.ang_diff(heading_a, heading_b) > heading_deg:
                continue
            lat_b, lon_b = pose_b
            bx, by = geo.latlon_to_local_xy(lat_b, lon_b, lat_a, lon_a)
            if math.hypot(bx, by) <= distance_m:
                union(i, j)
    roots = [find(i) for i in range(len(records))]
    remap = {}
    labels = []
    for root in roots:
        labels.append(remap.setdefault(root, len(remap)))
    return labels


def building_names(raw_path: Path) -> dict[str, str]:
    if not raw_path.is_file():
        return {}
    raw = load_json(raw_path)
    names = {}
    for element in (raw.get("osm_raw") or {}).get("elements") or []:
        tags = element.get("tags") or {}
        if "building" not in tags:
            continue
        label = tags.get("name") or ""
        number = tags.get("addr:housenumber")
        text = " ".join(part for part in (number, label) if part) or tags.get("building") or ""
        names[f"{element.get('type')}/{element.get('id')}"] = text
    return names


def font(size: int):
    for candidate in ("/System/Library/Fonts/Supplemental/Arial.ttf",
                      "/System/Library/Fonts/Helvetica.ttc"):
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def contact_sheet(selected: list[dict], thumbs: Path, dest: Path) -> None:
    cols = 4
    thumb_w, thumb_h, label_h, pad = 460, 280, 92, 10
    rows = math.ceil(len(selected) / cols) if selected else 1
    canvas = Image.new("RGB", (pad + cols * (thumb_w + pad), pad + rows * (thumb_h + label_h + pad)), (245, 246, 248))
    draw = ImageDraw.Draw(canvas)
    body = font(16)
    for index, img in enumerate(selected):
        col, row = index % cols, index // cols
        x = pad + col * (thumb_w + pad)
        y = pad + row * (thumb_h + label_h + pad)
        draw.rectangle((x, y, x + thumb_w, y + thumb_h + label_h), fill=(255, 255, 255), outline=(210, 214, 220))
        path = thumbs / f"{img['id']}.jpg"
        if path.is_file():
            with Image.open(path) as photo:
                photo = photo.convert("RGB")
                photo.thumbnail((thumb_w - 8, thumb_h - 8), Image.Resampling.LANCZOS)
                canvas.paste(photo, (x + 4, y + 4))
        sel = img["selection"]
        lines = [
            f"#{sel['selected_rank']}  {img['id']}",
            f"score={sel['final_score']:.3f}  bldg={sel['building_frac']:.2f}",
            f"veg={sel['vegetation_frac']:.2f} ctr={sel['center_vegetation_frac']:.2f}",
            f"osm={sel['osm_id']}",
        ]
        for line_i, line in enumerate(lines):
            draw.text((x + 8, y + thumb_h + 4 + line_i * 21), line[:58], fill=(20, 20, 20), font=body)
    dest.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(dest, quality=90)


def gallery(selected: list[dict], dest: Path, names: dict[str, str], dedup_note: str) -> None:
    cards = []
    for img in selected:
        sel = img["selection"]
        image_id = str(img["id"])
        pose = latlon(img)
        lat = f"{pose[0]:.6f}" if pose else "n/a"
        lon = f"{pose[1]:.6f}" if pose else "n/a"
        link = f"https://www.mapillary.com/app/?pKey={escape(image_id)}&focus=photo"
        name = names.get(sel["osm_id"], "")
        cards.append(f"""
        <article class="card">
          <img src="thumbs/{escape(image_id)}.jpg" alt="Mapillary {escape(image_id)}">
          <div class="meta">
            <h2>#{sel['selected_rank']} · ID {escape(image_id)}</h2>
            <p class="reason">{escape(sel['keep_reason'])}</p>
            <p><a href="{link}">Open on Mapillary</a></p>
            <dl>
              <dt>Final score</dt><dd>{sel['final_score']:.4f}</dd>
              <dt>Building fraction</dt><dd>{sel['building_frac']:.3f}</dd>
              <dt>Vegetation fraction</dt><dd>{sel['vegetation_frac']:.3f}</dd>
              <dt>Center vegetation</dt><dd>{sel['center_vegetation_frac']:.3f}</dd>
              <dt>Geometric score</dt><dd>{sel['geometric_score']:.4f}</dd>
              <dt>Sharpness (norm)</dt><dd>{sel['sharpness_normalized']:.3f}</dd>
              <dt>OSM building</dt><dd>{escape(sel['osm_id'])} {escape(name)}</dd>
              <dt>Latitude</dt><dd>{lat}</dd>
              <dt>Longitude</dt><dd>{lon}</dd>
              <dt>Compass</dt><dd>{sel['compass_angle'] if sel['compass_angle'] is not None else 'n/a'}</dd>
              <dt>Duplicate cluster</dt><dd>{sel['duplicate_cluster_id']} ({sel['cluster_size']} frames)</dd>
            </dl>
          </div>
        </article>""")
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Thompson Street best building views</title>
  <style>
    body {{ font-family: Georgia, serif; margin: 0; background: #f4f1ea; color: #1d1a16; }}
    header {{ padding: 28px 32px 8px; max-width: 900px; }}
    .gallery {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(420px, 1fr)); gap: 18px; padding: 18px 32px 40px; }}
    .card {{ background: white; border-radius: 12px; overflow: hidden; box-shadow: 0 1px 6px rgba(0,0,0,.08); }}
    img {{ width: 100%; height: 260px; object-fit: cover; display: block; background: #ddd; }}
    .meta {{ padding: 14px 16px 18px; }}
    h2 {{ font-size: 18px; margin: 0 0 6px; }}
    .reason {{ font-family: ui-sans-serif, system-ui, sans-serif; color: #333; }}
    dl {{ display: grid; grid-template-columns: 160px 1fr; gap: 4px 10px; margin: 10px 0 0; font-family: ui-sans-serif, system-ui, sans-serif; font-size: 14px; }}
    dt {{ color: #666; }} dd {{ margin: 0; }}
    a {{ color: #0b57d0; }}
  </style>
</head>
<body>
  <header>
    <h1>Best building views — {len(selected)} images that passed</h1>
    <p>Every geometrically accepted photo that survived the configured filters. There is no top-N cutoff. Score weights building coverage most, then low center vegetation, low overall vegetation, geometric score, and a small sharpness term. {escape(dedup_note)} At most 5 images are kept per OSM building. Rank is score order within this passing set.</p>
  </header>
  <section class="gallery">{''.join(cards)}</section>
</body>
</html>"""
    dest.write_text(html, encoding="utf-8")


def public_record(img: dict) -> dict:
    """Drop the selection block's internals that are recomputed; keep source metadata."""
    record = {key: value for key, value in img.items() if key != "selection"}
    record["selection"] = img["selection"]
    return record


def main():
    parser = argparse.ArgumentParser(description="Select best building views from geometric accepts")
    parser.add_argument("--geometric", type=Path, default=ROOT / "data/outputs/thompson_st_test/geometric/accepted.json")
    parser.add_argument("--vegetation", type=Path, default=ROOT / "data/outputs/thompson_st_test/vegetation/accepted.json")
    parser.add_argument("--raw", type=Path, default=ROOT / "data/outputs/thompson_st_test/raw/area_fetch.json")
    parser.add_argument("--thumb-cache", type=Path, default=ROOT / "data/outputs/thompson_st_test/visualization/thumbs")
    parser.add_argument("--out", type=Path, default=ROOT / "data/outputs/thompson_st_test/best_building_views")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/default.yaml")
    parser.add_argument("--max-per-building", type=int, default=None)
    parser.add_argument("--dedup", action=argparse.BooleanOptionalAction, default=None,
                        help="Override filter.dedup.enabled from the config")
    parser.add_argument("--dedup-distance-m", type=float, default=None)
    parser.add_argument("--dedup-heading-deg", type=float, default=None)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8")) if args.config.is_file() else {}
    dedup_cfg = ((config.get("filter") or {}).get("dedup") or {})
    dedup_enabled = dedup_cfg.get("enabled", True) if args.dedup is None else args.dedup
    dedup_distance_m = args.dedup_distance_m if args.dedup_distance_m is not None else float(dedup_cfg.get("distance_m", DUP_DISTANCE_M))
    dedup_heading_deg = args.dedup_heading_deg if args.dedup_heading_deg is not None else float(dedup_cfg.get("heading_deg", DUP_HEADING_DEG))
    max_per_building = args.max_per_building if args.max_per_building is not None else MAX_PER_BUILDING

    geometric = load_json(args.geometric)
    if not isinstance(geometric, list):
        raise ValueError("Expected geometric accepted JSON to be a list")
    # Vegetation file is used only to confirm the geometric set is a subset.
    vegetation = load_json(args.vegetation)
    veg_ids = {str(img.get("id")) for img in vegetation}
    missing_veg = [str(img.get("id")) for img in geometric if str(img.get("id")) not in veg_ids]
    if missing_veg:
        print(f"warning: {len(missing_veg)} geometric images were not in vegetation accepted")

    thumbs = args.out / "thumbs"
    thumbs.mkdir(parents=True, exist_ok=True)
    with requests.Session() as session:
        for img in geometric:
            cached = args.thumb_cache / f"{img['id']}.jpg"
            dest = thumbs / f"{img['id']}.jpg"
            if cached.is_file() and not dest.is_file():
                dest.write_bytes(cached.read_bytes())
            elif not dest.is_file():
                download_image(img, session, dest)

    sharp_values = [sharpness(thumbs / f"{img['id']}.jpg") for img in geometric]
    known = [value for value in sharp_values if value is not None]
    sharp_min = min(known) if known else 0.0
    sharp_max = max(known) if known else 1.0
    span = sharp_max - sharp_min

    ranked = []
    for img, sharp in zip(geometric, sharp_values):
        visual = img.get("visual_filter") or {}
        metrics = img.get("metrics") or {}
        building = frac(visual.get("building_frac"))
        vegetation_frac = frac(visual.get("vegetation_frac"))
        center = frac(visual.get("center_vegetation_frac"))
        geometric_score = frac(metrics.get("score"))
        if sharp is None or span <= 0:
            sharp_norm = 0.0 if sharp is None else 1.0
        else:
            sharp_norm = (sharp - sharp_min) / span
        final_score = (
            W_BUILDING * building
            + W_CENTER_VEG * (1.0 - center)
            + W_VEG * (1.0 - vegetation_frac)
            + W_GEOM * geometric_score
            + W_SHARP * sharp_norm
        )
        record = dict(img)
        record["selection"] = {
            "final_score": final_score,
            "building_frac": building,
            "vegetation_frac": vegetation_frac,
            "center_vegetation_frac": center,
            "geometric_score": geometric_score,
            "sharpness": sharp,
            "sharpness_normalized": sharp_norm,
            "osm_id": metrics.get("best_osm_id"),
            "compass_angle": heading_of(img),
            "duplicate_cluster_id": None,
            "cluster_size": None,
            "removed_as_duplicate": False,
            "removed_by_building_cap": False,
            "kept_after_filters": False,
            "selected_rank": None,
            "keep_reason": None,
        }
        ranked.append(record)

    if dedup_enabled:
        labels = cluster_duplicates(ranked, dedup_distance_m, dedup_heading_deg)
    else:
        labels = list(range(len(ranked)))
    groups = defaultdict(list)
    for index, label in enumerate(labels):
        ranked[index]["selection"]["duplicate_cluster_id"] = label
        groups[label].append(index)
    for label, members in groups.items():
        for index in members:
            ranked[index]["selection"]["cluster_size"] = len(members)
        winner = max(members, key=lambda i: ranked[i]["selection"]["final_score"])
        for index in members:
            if index != winner:
                ranked[index]["selection"]["removed_as_duplicate"] = True

    survivors = [img for img in ranked if not img["selection"]["removed_as_duplicate"]]
    survivors.sort(key=lambda img: img["selection"]["final_score"], reverse=True)
    counts = Counter()
    kept = []
    for img in survivors:
        osm_id = img["selection"]["osm_id"] or "unknown"
        if counts[osm_id] >= max_per_building:
            img["selection"]["removed_by_building_cap"] = True
            continue
        counts[osm_id] += 1
        img["selection"]["kept_after_filters"] = True
        kept.append(img)

    # Reasons for images that survived both filters.
    post_dedup_counts = Counter((img["selection"]["osm_id"] or "unknown") for img in survivors)
    for img in kept:
        osm_id = img["selection"]["osm_id"] or "unknown"
        if img["selection"]["cluster_size"] > 1:
            img["selection"]["keep_reason"] = "highest score in duplicate cluster"
        elif post_dedup_counts[osm_id] > max_per_building:
            img["selection"]["keep_reason"] = f"best remaining view for building {osm_id}"
        else:
            img["selection"]["keep_reason"] = f"best remaining view for building {osm_id}"

    ranked.sort(key=lambda img: img["selection"]["final_score"], reverse=True)
    for rank, img in enumerate(ranked, start=1):
        img["selection"]["score_rank"] = rank
    for rank, img in enumerate(kept, start=1):
        img["selection"]["selected_rank"] = rank

    names = building_names(args.raw)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    passing = [public_record(img) for img in kept]
    (out / "final_ranked_all.json").write_text(json.dumps([public_record(img) for img in ranked], indent=2), encoding="utf-8")
    (out / "final_selected.json").write_text(json.dumps(passing, indent=2), encoding="utf-8")
    # Older filenames now hold the same full passing set. Nothing is dropped to make a top-20 or top-40.
    (out / "final_selected_top20.json").write_text(json.dumps(passing, indent=2), encoding="utf-8")
    (out / "final_selected_top40.json").write_text(json.dumps(passing, indent=2), encoding="utf-8")
    if dedup_enabled:
        dedup_note = (f"Near-duplicate removal is on: frames within {dedup_distance_m:g} m and "
                      f"{dedup_heading_deg:g}° keep only the highest score.")
    else:
        dedup_note = "Near-duplicate removal is off, so every geometrically accepted frame stays eligible."
    gallery(kept, out / "final_selected_gallery.html", names, dedup_note)
    contact_sheet(kept, thumbs, out / "final_selected_contact_sheet.jpg")

    input_counts = Counter((img.get("metrics") or {}).get("best_osm_id") for img in geometric)
    final_counts = Counter(img["selection"]["osm_id"] for img in kept)
    kept_counts = Counter(img["selection"]["osm_id"] for img in kept)
    n_clusters = len(groups)
    multi = sum(1 for members in groups.values() if len(members) > 1)
    removed_dup = sum(1 for img in ranked if img["selection"]["removed_as_duplicate"])
    removed_cap = sum(1 for img in ranked if img["selection"]["removed_by_building_cap"])

    def fmt_counts(counter: Counter) -> str:
        lines = []
        for osm_id, count in counter.most_common():
            lines.append(f"- {osm_id} ({names.get(osm_id, 'unnamed')}): {count}")
        return "\n".join(lines)

    summary = f"""# Best building views — summary

## Counts

- Input candidate images (geometric accepts): {len(geometric)}
- Vegetation-accepted ids missing from that set: {len(missing_veg)}
- Duplicate clusters (connected components): {n_clusters}
- Clusters with more than one image: {multi}
- Images removed as near-duplicates: {removed_dup}
- Images remaining after deduplication: {len(survivors)}
- Images removed by the per-building cap of {max_per_building}: {removed_cap}
- Images remaining after deduplication and the building cap: {len(kept)}
- Final selected count: {len(kept)} (every image that passed; no top-20 or top-40 cutoff)

## Score

```
final_score =
    0.50 * building_fraction
  + 0.20 * (1 - center_vegetation_fraction)
  + 0.10 * (1 - vegetation_fraction)
  + 0.15 * geometric_score
  + 0.05 * sharpness_normalized
```

Sharpness is the variance of the Laplacian on the cached thumbnail, min-max normalized across the 90 candidates. Geometric score is the existing filter score, not rescaled.

Near-duplicates: {"enabled" if dedup_enabled else "disabled"} in `configs/default.yaml` under `filter.dedup`. When enabled, two images are linked when they are within {dedup_distance_m:g} m and {dedup_heading_deg:g} degrees of heading. Each cluster keeps its highest final score.

Building cap: after deduplication, images are taken in score order and a building is skipped once it already has {max_per_building} kept images.

## Buildings in the original geometric set

{fmt_counts(input_counts)}

## Buildings in the post-cap pool

{fmt_counts(kept_counts)}

## Buildings in the final selected set

{fmt_counts(final_counts)}

## Limitations

This selector rewards a large, lightly vegetated building region plus the existing geometric score. It does not measure façade occlusion, camera registration, or whether neighboring photos overlap enough for a reconstruction. The 8 m / 20° rule only removes nearby frames with similar headings. The five-image cap can drop a strong sixth view of a well-covered building in favor of a weaker view of a different building. Sharpness is computed on thumbnails, so it is only a rough tie-breaker.
"""
    (out / "summary.md").write_text(summary, encoding="utf-8")
    print(json.dumps({
        "input": len(geometric),
        "clusters": n_clusters,
        "multi_clusters": multi,
        "removed_duplicates": removed_dup,
        "after_dedup": len(survivors),
        "removed_cap": removed_cap,
        "kept": len(kept),
        "selected": len(kept),
        "selected_ids": [str(img["id"]) for img in kept],
        "selected_buildings": dict(final_counts),
    }, indent=2))


if __name__ == "__main__":
    main()
