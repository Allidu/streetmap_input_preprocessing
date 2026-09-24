import argparse
import json
import os
from html import escape


def load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def get_image_url(img: dict) -> str | None:
    # Mapillary field you are currently requesting
    return img.get("thumb_original_url")


def get_lat_lon(img: dict) -> tuple[str, str]:
    geom = img.get("computed_geometry", {})
    coords = geom.get("coordinates", [])

    if isinstance(coords, list) and len(coords) >= 2:
        lon, lat = coords[0], coords[1]
        return str(lat), str(lon)

    return "N/A", "N/A"


def get_compass_angle(img: dict) -> str:
    angle = img.get("computed_compass_angle")
    return "N/A" if angle is None else str(angle)


def get_captured_at(img: dict) -> str:
    captured = img.get("captured_at")
    return "N/A" if captured is None else str(captured)


def build_html(data: dict) -> str:
    mode = data.get("mode", "unknown")
    target_building = data.get("target_building", {})
    mapillary_images = data.get("mapillary", [])

    building_info = ""
    if target_building:
        building_info = f"""
        <div class="meta-block">
          <div><strong>Target Building OSM Type:</strong> {escape(str(target_building.get("osm_type", "N/A")))}</div>
          <div><strong>Target Building OSM ID:</strong> {escape(str(target_building.get("osm_id", "N/A")))}</div>
          <div><strong>Centroid:</strong> {escape(str(target_building.get("centroid", {})))}</div>
        </div>
        """

    cards = []
    for i, img in enumerate(mapillary_images, start=1):
        image_id = img.get("id", "N/A")
        image_url = get_image_url(img)
        lat, lon = get_lat_lon(img)
        angle = get_compass_angle(img)
        captured_at = get_captured_at(img)

        if image_url:
            image_html = f'<img src="{escape(image_url)}" alt="Mapillary image {escape(str(image_id))}">'
        else:
            image_html = '<div class="missing-image">No thumbnail URL available</div>'

        card = f"""
        <div class="card">
          <div class="img-wrap">
            {image_html}
          </div>
          <div class="card-body">
            <div><strong>Index:</strong> {i}</div>
            <div><strong>Image ID:</strong> {escape(str(image_id))}</div>
            <div><strong>Captured At:</strong> {escape(captured_at)}</div>
            <div><strong>Compass Angle:</strong> {escape(angle)}</div>
            <div><strong>Lat:</strong> {escape(lat)}</div>
            <div><strong>Lon:</strong> {escape(lon)}</div>
          </div>
        </div>
        """
        cards.append(card)

    cards_html = "\n".join(cards)

    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Mapillary Gallery</title>
  <style>
    body {{
      font-family: Arial, sans-serif;
      margin: 24px;
      background: #f7f7f7;
      color: #222;
    }}
    h1, h2 {{
      margin-bottom: 8px;
    }}
    .meta-block {{
      margin-bottom: 24px;
      padding: 12px 16px;
      background: white;
      border-radius: 10px;
      box-shadow: 0 1px 4px rgba(0,0,0,0.08);
    }}
    .gallery {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
      gap: 18px;
    }}
    .card {{
      background: white;
      border-radius: 12px;
      overflow: hidden;
      box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }}
    .img-wrap {{
      width: 100%;
      height: 240px;
      background: #eee;
      display: flex;
      align-items: center;
      justify-content: center;
    }}
    img {{
      width: 100%;
      height: 100%;
      object-fit: cover;
      display: block;
    }}
    .card-body {{
      padding: 12px 14px;
      line-height: 1.5;
      font-size: 14px;
    }}
    .missing-image {{
      color: #777;
      font-style: italic;
    }}
  </style>
</head>
<body>
  <h1>Mapillary Image Gallery</h1>
  <div class="meta-block">
    <div><strong>Mode:</strong> {escape(str(mode))}</div>
    <div><strong>Number of Images:</strong> {len(mapillary_images)}</div>
  </div>

  {building_info}

  <h2>Images</h2>
  <div class="gallery">
    {cards_html}
  </div>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(
        description="Generate an HTML gallery from a building_fetch.json file."
    )
    parser.add_argument("--input", required=True, help="Path to building_fetch.json")
    parser.add_argument(
        "--output",
        default="data/outputs/mapillary_gallery.html",
        help="Output HTML file path",
    )
    args = parser.parse_args()

    data = load_json(args.input)
    html = build_html(data)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Saved gallery to {args.output}")


if __name__ == "__main__":
    main()