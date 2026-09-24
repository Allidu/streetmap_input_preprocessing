import json
import argparse
from pathlib import Path
from html import escape


def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


def get_score(item):
    return item.get("metrics", {}).get("score", float("-inf"))


def format_value(val, decimals=3):
    if isinstance(val, (int, float)):
        return f"{val:.{decimals}f}"
    return str(val)


def build_card(item):
    image_id = item.get("id", "N/A")
    image_url = item.get("thumb_original_url", "")
    captured_at = item.get("captured_at", "N/A")
    score = item.get("metrics", {}).get("score", "N/A")
    d_min = item.get("metrics", {}).get("d_min_m", "N/A")
    theta_min = item.get("metrics", {}).get("theta_min_deg", "N/A")
    d_eff = item.get("metrics", {}).get("d_eff_m", "N/A")
    compass = item.get("computed_compass_angle", "N/A")

    lat = "N/A"
    lon = "N/A"
    coords = item.get("computed_geometry", {}).get("coordinates", [])
    if isinstance(coords, list) and len(coords) == 2:
        lon, lat = coords[0], coords[1]

    return f"""
    <div class="card">
        <div class="image-wrap">
            <img src="{escape(str(image_url))}" alt="Image {escape(str(image_id))}" loading="lazy">
        </div>
        <div class="content">
            <div class="top-row">
                <h3>ID: {escape(str(image_id))}</h3>
                <div class="score">Score: {format_value(score, 6)}</div>
            </div>

            <div class="meta-grid">
                <div><strong>Captured at:</strong> {escape(str(captured_at))}</div>
                <div><strong>Compass angle:</strong> {format_value(compass)}</div>
                <div><strong>Latitude:</strong> {format_value(lat, 6)}</div>
                <div><strong>Longitude:</strong> {format_value(lon, 6)}</div>
                <div><strong>d_min_m:</strong> {format_value(d_min)}</div>
                <div><strong>theta_min_deg:</strong> {format_value(theta_min)}</div>
                <div><strong>d_eff_m:</strong> {format_value(d_eff)}</div>
            </div>
        </div>
    </div>
    """


def generate_html(items, title="Ranked Image Gallery"):
    cards_html = "\n".join(build_card(item) for item in items)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{escape(title)}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 24px;
            background: #f5f7fb;
            color: #222;
        }}

        h1 {{
            margin-bottom: 8px;
        }}

        .subtitle {{
            margin-bottom: 24px;
            color: #555;
        }}

        .gallery {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(380px, 1fr));
            gap: 20px;
        }}

        .card {{
            background: white;
            border-radius: 14px;
            overflow: hidden;
            box-shadow: 0 4px 14px rgba(0, 0, 0, 0.08);
            display: flex;
            flex-direction: column;
        }}

        .image-wrap {{
            width: 100%;
            background: #ddd;
        }}

        .image-wrap img {{
            width: 100%;
            height: 280px;
            object-fit: cover;
            display: block;
        }}

        .content {{
            padding: 16px;
        }}

        .top-row {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 12px;
            margin-bottom: 12px;
        }}

        .top-row h3 {{
            margin: 0;
            font-size: 16px;
            word-break: break-all;
        }}

        .score {{
            font-weight: bold;
            color: #0b5;
            white-space: nowrap;
        }}

        .meta-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 8px 16px;
            font-size: 14px;
            line-height: 1.4;
        }}

        @media (max-width: 700px) {{
            .gallery {{
                grid-template-columns: 1fr;
            }}

            .meta-grid {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <h1>{escape(title)}</h1>
    <div class="subtitle">
        Images sorted by descending <code>metrics.score</code>
    </div>

    <div class="gallery">
        {cards_html}
    </div>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(description="Visualize ranked JSON images as HTML.")
    parser.add_argument("--input", required=True, help="Path to input JSON file")
    parser.add_argument("--output",  default="data/outputs/filter_gallery.html", help="Path to output HTML file")
    parser.add_argument("--title", default="Ranked Image Gallery", help="HTML page title")
    args = parser.parse_args()

    data = load_json(args.input)

    if not isinstance(data, list):
        raise ValueError("Expected JSON file to contain a list of image entries.")

    sorted_data = sorted(data, key=get_score, reverse=True)

    html = generate_html(sorted_data, title=args.title)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")

    print(f"Saved HTML gallery to {output_path}")


if __name__ == "__main__":
    main()