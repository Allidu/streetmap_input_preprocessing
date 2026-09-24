#!/usr/bin/env python3
"""Build Thompson Street evaluation artifacts from completed pipeline outputs.

Does not rerun segmentation. Downloads photographs from the URLs already stored
on each Mapillary record so galleries do not depend on a later page view.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from collections import Counter
from html import escape
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageFont

import filter_metadata as geo


ROOT = Path(__file__).resolve().parents[1]
ERROR_REASON = "image_or_segmentation_error"


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


def visual(img: dict) -> dict:
    return img.get("visual_filter") or {}


def metrics(img: dict) -> dict:
    return img.get("metrics") or {}


def spread(records: list[dict], count: int) -> list[dict]:
    ordered = sorted(records, key=lambda img: (latlon(img) or (0.0, 0.0))[0])
    if len(ordered) <= count:
        return ordered
    if count <= 1:
        return [ordered[len(ordered) // 2]]
    indexes = [round(i * (len(ordered) - 1) / (count - 1)) for i in range(count)]
    chosen = []
    seen = set()
    for index in indexes:
        image_id = str(ordered[index].get("id"))
        if image_id in seen:
            continue
        seen.add(image_id)
        chosen.append(ordered[index])
    return chosen


def download_image(img: dict, session: requests.Session, dest: Path, max_edge: int = 960) -> bool:
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
            image.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
            image.save(dest, quality=85)
        raw.unlink(missing_ok=True)
        return True
    except Exception as exc:
        print(f"download_failed {img.get('id')} {type(exc).__name__}")
        return False


def font(size: int) -> ImageFont.ImageFont:
    for candidate in ("/System/Library/Fonts/Supplemental/Arial.ttf",
                      "/Library/Fonts/Arial.ttf",
                      "/System/Library/Fonts/Helvetica.ttc"):
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def comparison_sheet(columns: list[tuple[str, list[tuple[dict, Path]]]], dest: Path) -> None:
    thumb_w, thumb_h = 420, 280
    label_h = 78
    header_h = 46
    pad = 12
    rows = max((len(items) for _, items in columns), default=0)
    width = pad + len(columns) * (thumb_w + pad)
    height = pad + header_h + rows * (thumb_h + label_h + pad)
    canvas = Image.new("RGB", (width, max(height, 80)), (245, 246, 248))
    draw = ImageDraw.Draw(canvas)
    title_font = font(22)
    body_font = font(15)
    for col, (title, items) in enumerate(columns):
        x = pad + col * (thumb_w + pad)
        draw.text((x, pad), title, fill=(20, 20, 20), font=title_font)
        for row in range(rows):
            y = pad + header_h + row * (thumb_h + label_h + pad)
            draw.rectangle((x, y, x + thumb_w, y + thumb_h + label_h), fill=(255, 255, 255), outline=(210, 214, 220))
            if row >= len(items):
                draw.text((x + 12, y + 12), "no image", fill=(120, 120, 120), font=body_font)
                continue
            img, path = items[row]
            if path.is_file():
                with Image.open(path) as photo:
                    photo = photo.convert("RGB")
                    photo.thumbnail((thumb_w - 8, thumb_h - 8), Image.Resampling.LANCZOS)
                    canvas.paste(photo, (x + 4, y + 4))
            vf = visual(img)
            gm = metrics(img)
            lines = [
                f"ID {img.get('id')}",
                f"bldg={num(vf.get('building_frac'))} veg={num(vf.get('vegetation_frac'))} ctr={num(vf.get('center_vegetation_frac'))}",
                f"score={num(gm.get('score'), 3)} osm={gm.get('best_osm_id') or '-'}",
            ]
            for i, line in enumerate(lines):
                draw.text((x + 8, y + thumb_h + 4 + i * 22), line[:62], fill=(20, 20, 20), font=body_font)
    dest.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(dest, quality=90)


def num(value, digits=2):
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return "n/a"


def write_gallery(items: list[dict], thumbs: Path, dest: Path) -> None:
    cards = []
    for img in sorted(items, key=lambda item: metrics(item).get("score") or -1, reverse=True):
        image_id = str(img.get("id"))
        local = thumbs / f"{image_id}.jpg"
        src = local.name if local.is_file() else ""
        lat_lon = latlon(img)
        lat = f"{lat_lon[0]:.6f}" if lat_lon else "n/a"
        lon = f"{lat_lon[1]:.6f}" if lat_lon else "n/a"
        vf = visual(img)
        gm = metrics(img)
        link = f"https://www.mapillary.com/app/?pKey={escape(image_id)}&focus=photo"
        image_html = f'<img src="thumbs/{escape(src)}" alt="Mapillary {escape(image_id)}">' if src else "<div class='missing'>Thumbnail unavailable</div>"
        cards.append(f"""
        <article class="card">
          <div class="photo">{image_html}</div>
          <div class="meta">
            <h2>ID {escape(image_id)}</h2>
            <p><a href="{link}">Open on Mapillary</a></p>
            <dl>
              <dt>Latitude</dt><dd>{lat}</dd>
              <dt>Longitude</dt><dd>{lon}</dd>
              <dt>Compass</dt><dd>{escape(num(heading_of(img), 1))}°</dd>
              <dt>Building fraction</dt><dd>{escape(num(vf.get('building_frac')))}</dd>
              <dt>Vegetation fraction</dt><dd>{escape(num(vf.get('vegetation_frac')))}</dd>
              <dt>Center vegetation</dt><dd>{escape(num(vf.get('center_vegetation_frac')))}</dd>
              <dt>Geometric score</dt><dd>{escape(num(gm.get('score'), 4))}</dd>
              <dt>OSM building</dt><dd>{escape(str(gm.get('best_osm_id') or 'n/a'))}</dd>
              <dt>Distance (m)</dt><dd>{escape(num(gm.get('d_min_m'), 1))}</dd>
              <dt>Min view angle</dt><dd>{escape(num(gm.get('theta_min_deg'), 1))}°</dd>
              <dt>Captured</dt><dd>{escape(str(img.get('captured_at', 'n/a')))}</dd>
            </dl>
          </div>
        </article>""")
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Thompson Street accepted photographs</title>
  <style>
    body {{ font-family: Georgia, serif; margin: 0; background: #f4f1ea; color: #1d1a16; }}
    header {{ padding: 28px 32px 8px; }}
    h1 {{ margin: 0 0 8px; font-weight: 600; }}
    .gallery {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(420px, 1fr)); gap: 18px; padding: 18px 32px 40px; }}
    .card {{ background: white; border-radius: 12px; overflow: hidden; box-shadow: 0 1px 6px rgba(0,0,0,.08); }}
    .photo img {{ width: 100%; height: 260px; object-fit: cover; display: block; background: #ddd; }}
    .meta {{ padding: 14px 16px 18px; }}
    h2 {{ font-size: 18px; margin: 0 0 6px; }}
    dl {{ display: grid; grid-template-columns: 160px 1fr; gap: 4px 10px; margin: 10px 0 0; font-family: ui-sans-serif, system-ui, sans-serif; font-size: 14px; }}
    dt {{ color: #666; }}
    dd {{ margin: 0; }}
    a {{ color: #0b57d0; }}
    .missing {{ height: 260px; display: flex; align-items: center; justify-content: center; background: #eee; }}
  </style>
</head>
<body>
  <header>
    <h1>Thompson Street — photographs accepted after vegetation and geometric filtering</h1>
    <p>{len(items)} images. Building fraction and vegetation fractions come from Mask2Former. Center vegetation is the fraction of the central 70% of the frame labeled vegetation, not a measured façade occlusion. Geometric score uses the existing distance and viewing-direction filter.</p>
  </header>
  <section class="gallery">{''.join(cards)}</section>
</body>
</html>"""
    dest.write_text(html, encoding="utf-8")


def destination_point(lat: float, lon: float, bearing_deg: float, distance_m: float) -> tuple[float, float]:
    mlat, mlon = geo.meters_per_degree(lat)
    rad = math.radians(bearing_deg)
    return lat + (distance_m * math.cos(rad) / mlat), lon + (distance_m * math.sin(rad) / mlon)


def write_map(raw: dict, groups: dict[str, list[dict]], dest: Path) -> None:
    south, west, north, east = raw["bbox_south_west_north_east"]
    buildings = []
    for building in geo.extract_osm_buildings(raw.get("osm_raw")):
        ring = [[lat, lon] for lat, lon in building["coords"]]
        buildings.append({"id": building["osm_id"], "ring": ring})
    cameras = []
    for status, records in groups.items():
        for img in records:
            pose = latlon(img)
            if pose is None:
                continue
            heading = heading_of(img)
            tip = destination_point(pose[0], pose[1], heading, 18.0) if heading is not None else None
            cameras.append({
                "id": str(img.get("id")),
                "status": status,
                "lat": pose[0],
                "lon": pose[1],
                "heading": heading,
                "tip": tip,
                "score": metrics(img).get("score"),
                "osm": metrics(img).get("best_osm_id"),
                "reason": visual(img).get("reason") or img.get("reason"),
            })
    payload = {
        "bbox": [[south, west], [north, east]],
        "center": [raw["center"]["lat"], raw["center"]["lon"]],
        "buildings": buildings,
        "cameras": cameras,
    }
    html = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Thompson Street geographic coverage</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
  <style>
    html, body, #map { height: 100%; margin: 0; }
    .legend { background: white; padding: 10px 12px; line-height: 1.45; font: 13px/1.4 sans-serif; }
    .swatch { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 6px; }
  </style>
</head>
<body>
  <div id="map"></div>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script>
    const data = __PAYLOAD__;
    const colors = {accepted: "#1b7f3a", veg_rejected: "#d35400", geo_rejected: "#2c3e90", error: "#888"};
    const map = L.map("map");
    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      maxZoom: 20,
      attribution: "&copy; OpenStreetMap"
    }).addTo(map);
    L.rectangle(data.bbox, {color: "#c0392b", weight: 2, fill: false}).addTo(map);
    L.circleMarker(data.center, {radius: 5, color: "#c0392b", fillOpacity: 1}).bindTooltip("search center").addTo(map);
    data.buildings.forEach(b => {
      L.polygon(b.ring, {color: "#555", weight: 1, fillColor: "#8d8d8d", fillOpacity: 0.25})
        .bindTooltip(b.id).addTo(map);
    });
    data.cameras.forEach(c => {
      const color = colors[c.status] || "#333";
      L.circleMarker([c.lat, c.lon], {radius: c.status === "accepted" ? 6 : 4, color, fillColor: color, fillOpacity: 0.9, weight: 1})
        .bindTooltip(`${c.status}<br>ID ${c.id}<br>heading ${c.heading}<br>score ${c.score}<br>osm ${c.osm}<br>${c.reason || ""}`)
        .addTo(map);
      if (c.tip) {
        L.polyline([[c.lat, c.lon], c.tip], {color, weight: c.status === "accepted" ? 2.5 : 1.2, opacity: 0.85}).addTo(map);
      }
    });
    map.fitBounds(data.bbox, {padding: [24, 24]});
    const legend = L.control({position: "bottomleft"});
    legend.onAdd = function() {
      const div = L.DomUtil.create("div", "legend");
      div.innerHTML = "<b>Thompson Street test</b><br>"
        + "<span class='swatch' style='background:#1b7f3a'></span>Final accepted<br>"
        + "<span class='swatch' style='background:#d35400'></span>Rejected by vegetation<br>"
        + "<span class='swatch' style='background:#2c3e90'></span>Rejected by geometry<br>"
        + "<span class='swatch' style='background:#888'></span>Download or segmentation error<br>"
        + "Lines show compass heading (~18 m). Gray polygons are OSM building footprints.";
      return div;
    };
    legend.addTo(map);
  </script>
</body>
</html>"""
    dest.write_text(html.replace("__PAYLOAD__", json.dumps(payload)), encoding="utf-8")


def near_duplicates(records: list[dict], distance_m=8.0, heading_deg=20.0) -> int:
    poses = []
    for img in records:
        pose = latlon(img)
        heading = heading_of(img)
        if pose and heading is not None:
            poses.append((img, pose, heading))
    flagged = set()
    for i, (img_a, (lat_a, lon_a), heading_a) in enumerate(poses):
        ax, ay = geo.latlon_to_local_xy(lat_a, lon_a, lat_a, lon_a)
        for img_b, (lat_b, lon_b), heading_b in poses[i + 1:]:
            bx, by = geo.latlon_to_local_xy(lat_b, lon_b, lat_a, lon_a)
            if math.hypot(bx - ax, by - ay) <= distance_m and geo.ang_diff(heading_a, heading_b) <= heading_deg:
                flagged.add(str(img_a.get("id")))
                flagged.add(str(img_b.get("id")))
    return len(flagged)


def write_summary(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Organize Thompson Street pipeline outputs and visuals")
    parser.add_argument("--raw", type=Path, default=ROOT / "data/raw/area_fetch.json")
    parser.add_argument("--veg-accepted", type=Path, default=ROOT / "data/intermediate/area_vegetation_accepted.json")
    parser.add_argument("--veg-rejected", type=Path, default=ROOT / "data/intermediate/area_vegetation_rejected.json")
    parser.add_argument("--veg-csv", type=Path, default=ROOT / "data/outputs/area_vegetation_scores.csv")
    parser.add_argument("--contact-sheet", type=Path, default=ROOT / "data/outputs/area_vegetation_contact_sheet.jpg")
    parser.add_argument("--geo-accepted", type=Path, default=ROOT / "data/filtered/area_filtered.json")
    parser.add_argument("--geo-rejected", type=Path, default=ROOT / "data/filtered/area_geometric_rejected.json")
    parser.add_argument("--out", type=Path, default=ROOT / "data/outputs/thompson_st_test")
    args = parser.parse_args()

    raw = load_json(args.raw)
    veg_area = load_json(args.veg_accepted)
    veg_rejected_all = load_json(args.veg_rejected)
    geo_accepted = load_json(args.geo_accepted)
    geo_rejected = load_json(args.geo_rejected)
    veg_ok = veg_area["mapillary"]
    errors = [img for img in veg_rejected_all if visual(img).get("reason") == ERROR_REASON]
    veg_no = [img for img in veg_rejected_all if visual(img).get("reason") != ERROR_REASON]

    out = args.out
    (out / "raw").mkdir(parents=True, exist_ok=True)
    (out / "vegetation").mkdir(parents=True, exist_ok=True)
    (out / "geometric").mkdir(parents=True, exist_ok=True)
    vis = out / "visualization"
    thumbs = vis / "thumbs"
    review = vis / "review"
    thumbs.mkdir(parents=True, exist_ok=True)
    review.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.raw, out / "raw" / "area_fetch.json")
    (out / "vegetation" / "accepted.json").write_text(json.dumps(veg_ok, indent=2), encoding="utf-8")
    (out / "vegetation" / "rejected.json").write_text(json.dumps(veg_no, indent=2), encoding="utf-8")
    (out / "vegetation" / "processing_errors.json").write_text(json.dumps(errors, indent=2), encoding="utf-8")
    shutil.copyfile(args.veg_csv, out / "vegetation" / "vegetation_scores.csv")
    if args.contact_sheet.is_file():
        shutil.copyfile(args.contact_sheet, out / "vegetation" / "vegetation_contact_sheet.jpg")
    (out / "geometric" / "accepted.json").write_text(json.dumps(geo_accepted, indent=2), encoding="utf-8")
    (out / "geometric" / "rejected.json").write_text(json.dumps(geo_rejected, indent=2), encoding="utf-8")

    comparison_columns = [
        ("1. Retrieved sample", spread(raw["mapillary"], 4)),
        ("2. After vegetation filter", spread(veg_ok, 4)),
        ("3. After geometric filter", spread(geo_accepted, 4)),
    ]
    needed = {}
    for _, records in comparison_columns:
        for img in records:
            needed[str(img["id"])] = img
    for img in geo_accepted:
        needed[str(img["id"])] = img
    for img in spread(veg_no, 8) + spread(geo_rejected, 6):
        needed[str(img["id"])] = img
    # Extremes used for the written evaluation.
    extremes = []
    if veg_no:
        extremes.append(min(veg_no, key=lambda img: visual(img).get("building_frac") or 1))
        extremes.append(max(veg_no, key=lambda img: visual(img).get("center_vegetation_frac") or 0))
        low_but_building = [img for img in veg_no if (visual(img).get("building_frac") or 0) >= 0.15]
        if low_but_building:
            extremes.append(max(low_but_building, key=lambda img: visual(img).get("center_vegetation_frac") or 0))
    if veg_ok:
        extremes.append(min(veg_ok, key=lambda img: visual(img).get("building_frac") or 1))
        extremes.append(max(veg_ok, key=lambda img: visual(img).get("center_vegetation_frac") or 0))
    if geo_accepted:
        extremes.append(max(geo_accepted, key=lambda img: metrics(img).get("theta_min_deg") or -1))
        extremes.append(min(geo_accepted, key=lambda img: metrics(img).get("score") or 1))
    if geo_rejected:
        facing = [img for img in geo_rejected if (metrics(img).get("theta_min_deg") or 999) <= 25 and (visual(img).get("building_frac") or 0) >= 0.15]
        extremes.append(max(facing or geo_rejected, key=lambda img: visual(img).get("building_frac") or 0))
    for img in extremes:
        needed[str(img["id"])] = img

    failed = []
    with requests.Session() as session:
        for image_id, img in needed.items():
            if not download_image(img, session, thumbs / f"{image_id}.jpg"):
                failed.append(image_id)

    comparison_sheet([
        (title, [(img, thumbs / f"{img.get('id')}.jpg") for img in records])
        for title, records in comparison_columns
    ], vis / "filtering_comparison.jpg")
    write_gallery(geo_accepted, thumbs, vis / "accepted_gallery.html")
    write_map(raw, {
        "accepted": geo_accepted,
        "veg_rejected": veg_no,
        "geo_rejected": geo_rejected,
        "error": errors,
    }, vis / "geographic_coverage.html")

    # Copy a few review stills under stable names for inspection.
    for img in extremes:
        src = thumbs / f"{img.get('id')}.jpg"
        if src.is_file():
            shutil.copyfile(src, review / f"{img.get('id')}.jpg")

    reason_counts = Counter(visual(img).get("reason") for img in veg_no)
    geo_reasons = Counter(img.get("reason") for img in geo_rejected)
    building_counts = Counter(metrics(img).get("best_osm_id") for img in geo_accepted)
    lats = [latlon(img)[0] for img in geo_accepted if latlon(img)]
    summary = {
        "retrieved": len(raw["mapillary"]),
        "unique_ids": len({str(img.get("id")) for img in raw["mapillary"]}),
        "gps": sum(latlon(img) is not None for img in raw["mapillary"]),
        "headings": sum(heading_of(img) is not None for img in raw["mapillary"]),
        "veg_accepted": len(veg_ok),
        "veg_rejected": len(veg_no),
        "errors": len(errors),
        "geo_accepted": len(geo_accepted),
        "geo_rejected": len(geo_rejected),
        "veg_reasons": dict(reason_counts),
        "geo_reasons": dict(geo_reasons),
        "building_counts": dict(building_counts),
        "accepted_lat_range": [min(lats), max(lats)] if lats else None,
        "near_duplicate_accepted": near_duplicates(geo_accepted),
        "download_failures": failed,
        "osm_buildings": len(geo.extract_osm_buildings(raw.get("osm_raw"))),
        "bbox": raw.get("bbox_south_west_north_east"),
        "center": raw.get("center"),
        "veg_summary": veg_area.get("visual_filter_summary"),
        "extremes": [{
            "id": str(img.get("id")),
            "veg_reason": visual(img).get("reason"),
            "building_frac": visual(img).get("building_frac"),
            "vegetation_frac": visual(img).get("vegetation_frac"),
            "center_vegetation_frac": visual(img).get("center_vegetation_frac"),
            "geo_reason": img.get("reason"),
            "score": metrics(img).get("score"),
            "osm": metrics(img).get("best_osm_id"),
            "theta_min_deg": metrics(img).get("theta_min_deg"),
            "d_min_m": metrics(img).get("d_min_m"),
            "heading": heading_of(img),
            "latlon": latlon(img),
        } for img in extremes],
    }
    write_summary(vis / "summary.json", summary)
    # Confirm CSV row count matches retrieved images.
    with args.veg_csv.open(encoding="utf-8") as handle:
        summary["csv_rows"] = sum(1 for _ in csv.DictReader(handle))
    write_summary(vis / "summary.json", summary)
    print(json.dumps({k: summary[k] for k in ("retrieved", "veg_accepted", "veg_rejected", "errors", "geo_accepted", "geo_rejected", "veg_reasons", "geo_reasons", "download_failures", "near_duplicate_accepted")}, indent=2))


if __name__ == "__main__":
    main()
