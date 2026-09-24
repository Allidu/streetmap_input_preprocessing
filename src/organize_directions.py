#!/usr/bin/env python3

import json
import argparse
from pathlib import Path


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def get_direction(heading):
    if heading is None:
        return "unknown"

    heading = heading % 360

    if heading >= 315 or heading < 45:
        return "north"
    elif heading < 135:
        return "east"
    elif heading < 225:
        return "south"
    else:
        return "west"


def get_score(item):
    return item.get("metrics", {}).get("score", -1)


def organize_by_direction(data):
    buckets = {
        "north": [],
        "east": [],
        "south": [],
        "west": [],
        "unknown": []
    }

    for item in data:
        heading = item.get("computed_compass_angle")
        direction = get_direction(heading)
        buckets[direction].append(item)

    # sort each bucket by descending score
    for k in buckets:
        buckets[k].sort(key=get_score, reverse=True)

    return buckets


def save_json(data, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ---------------- HTML (optional but useful) ---------------- #

def build_html(buckets, title="Directional Image Ranking"):
    def build_section(name, items):
        cards = []
        for i, item in enumerate(items):
            url = item.get("thumb_original_url", "")
            score = item.get("metrics", {}).get("score", 0)
            img_id = item.get("id")

            cards.append(f"""
            <div class="card">
                <img src="{url}">
                <div class="info">
                    <div><b>Rank:</b> {i+1}</div>
                    <div><b>ID:</b> {img_id}</div>
                    <div><b>Score:</b> {score:.4f}</div>
                </div>
            </div>
            """)

        return f"""
        <h2>{name.upper()} ({len(items)})</h2>
        <div class="grid">
            {''.join(cards)}
        </div>
        """

    sections = "".join(build_section(k, v) for k, v in buckets.items())

    return f"""
    <html>
    <head>
    <style>
        body {{
            font-family: Arial;
            background: #f5f5f5;
            padding: 20px;
        }}
        h2 {{
            margin-top: 40px;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
            gap: 16px;
        }}
        .card {{
            background: white;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .card img {{
            width: 100%;
            height: 180px;
            object-fit: cover;
        }}
        .info {{
            padding: 10px;
            font-size: 14px;
        }}
    </style>
    </head>
    <body>
        <h1>{title}</h1>
        {sections}
    </body>
    </html>
    """


def main():
    parser = argparse.ArgumentParser(description="Organize accepted images by direction.")
    parser.add_argument("--input", required=True, help="accepted_target_building.json")
    parser.add_argument("--output_json", required=True, help="output grouped json")
    parser.add_argument("--output_html", default=None, help="optional HTML output for debug")

    args = parser.parse_args()

    data = load_json(args.input)

    if not isinstance(data, list):
        raise ValueError("Expected input JSON to be a list.")

    buckets = organize_by_direction(data)

    save_json(buckets, args.output_json)
    print(f"Saved JSON to {args.output_json}")

    if args.output_html:
        html = build_html(buckets)
        Path(args.output_html).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_html).write_text(html)
        print(f"Saved HTML to {args.output_html}")


if __name__ == "__main__":
    main()