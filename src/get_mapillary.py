import argparse
import json
import os
import time
import requests

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
MAPILLARY_URL = "https://graph.mapillary.com/images"
# Overpass rejects the default python-requests User-Agent with HTTP 406.
REQUEST_HEADERS = {
    "User-Agent": "area-veg-integration/1.0 (academic street-level imagery research)",
}


def bbox_from_geometry_lat_lon(geometry_lat_lon, pad_deg):
    lats = [pt[0] for pt in geometry_lat_lon]
    lons = [pt[1] for pt in geometry_lat_lon]

    return (
        min(lats) - pad_deg,
        min(lons) - pad_deg,
        max(lats) + pad_deg,
        max(lons) + pad_deg,
    )


def merge_bboxes(b1, b2):
    s1, w1, n1, e1 = b1
    s2, w2, n2, e2 = b2
    return (
        min(s1, s2),
        min(w1, w2),
        max(n1, n2),
        max(e1, e2),
    )

def bbox_from_center(lat, lon, buffer_deg):
    return (
        lat - buffer_deg,
        lon - buffer_deg,
        lat + buffer_deg,
        lon + buffer_deg,
    )


def parse_coord(coord_str):
    lat, lon = map(float, coord_str.split(","))
    return lat, lon


def overpass_query_buildings(bbox):
    s, w, n, e = bbox
    return f"""
[out:json][timeout:25];
(
  way["building"]({s},{w},{n},{e});
  relation["building"]({s},{w},{n},{e});
);
out body;
>;
out skel qt;
"""


def fetch_osm(bbox, max_retries=3, sleep_seconds=3):
    last_err = None

    for attempt in range(max_retries):
        try:
            r = requests.post(
                OVERPASS_URL,
                data={"data": overpass_query_buildings(bbox)},
                headers=REQUEST_HEADERS,
                timeout=90,
            )
            r.raise_for_status()
            return r.json()

        except requests.exceptions.RequestException as e:
            last_err = e
            print(f"Overpass request failed (attempt {attempt + 1}/{max_retries}): {e}")
            time.sleep(sleep_seconds)

    raise last_err

import time

def fetch_mapillary(bbox, token, limit=200, max_retries=3, sleep_seconds=2):
    s, w, n, e = bbox
    bbox_str = f"{w},{s},{e},{n}"  # west,south,east,north

    params = {
        "bbox": bbox_str,
        "fields": "id,thumb_original_url,computed_geometry,computed_compass_angle,captured_at,camera_parameters",
        "access_token": token,
        "limit": limit,
    }

    all_images = []
    next_url = None
    page_num = 1

    while True:
        last_err = None

        for attempt in range(max_retries):
            try:
                if next_url is None:
                    r = requests.get(MAPILLARY_URL, params=params, timeout=60)
                else:
                    r = requests.get(next_url, timeout=60)

                r.raise_for_status()
                payload = r.json()
                break

            except requests.exceptions.RequestException as e:
                last_err = e
                print(
                    f"Mapillary request failed on page {page_num} "
                    f"(attempt {attempt + 1}/{max_retries}): {e}"
                )
                if attempt < max_retries - 1:
                    time.sleep(sleep_seconds)
        else:
            raise last_err

        page_images = payload.get("data", [])
        all_images.extend(page_images)

        paging = payload.get("paging", {})
        next_url = paging.get("next")

        print(f"Fetched page {page_num}: {len(page_images)} images")
        page_num += 1

        if not next_url:
            break

    print(f"Total Mapillary images fetched: {len(all_images)}")
    return all_images

def run_area_mode(center, buffer_deg, token):
    lat, lon = center
    bbox = bbox_from_center(lat, lon, buffer_deg)

    result = {
        "mode": "area",
        "center": {"lat": lat, "lon": lon},
        "bbox_south_west_north_east": bbox,
        "osm_raw": fetch_osm(bbox),
        "mapillary": fetch_mapillary(bbox, token),
    }

    time.sleep(0.25)
    return result


import json

def run_building_mode(building_file, token):
    with open(building_file, "r") as f:
        building_data = json.load(f)

    target_building = building_data["target_building"]
    centroid = target_building["centroid"]
    geometry_lat_lon = target_building["geometry_lat_lon"]

    centroid_lat = centroid["lat"]
    centroid_lon = centroid["lon"]

    # Small bbox for nearby OSM context
    OSM_BUFFER_DEG = 0.0005
    osm_bbox = bbox_from_center(centroid_lat, centroid_lon, OSM_BUFFER_DEG)

    # Larger bbox for Mapillary fetch, based on full building footprint
    MAPILLARY_PAD_DEG = 0.0004
    mapillary_bbox = bbox_from_geometry_lat_lon(
        geometry_lat_lon,
        pad_deg=MAPILLARY_PAD_DEG,
    )

    combined_bbox = merge_bboxes(osm_bbox, mapillary_bbox)

    result = {
        "mode": "building",
        "target_building": target_building,
        "osm_bbox_south_west_north_east": osm_bbox,
        "mapillary_bbox_south_west_north_east": mapillary_bbox,
        "bbox_south_west_north_east": combined_bbox,
        "osm_raw": fetch_osm(osm_bbox),
        "mapillary": fetch_mapillary(mapillary_bbox, token, limit=200),
    }

    time.sleep(0.25)
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Fetch OSM buildings and nearby Mapillary images."
    )

    parser.add_argument(
        "--mode",
        choices=["area", "building"],
        required=True,
        help="Whether to search an area around one center or a single target building.",
    )

    parser.add_argument(
        "--center",
        help="Area mode only: single center coordinate as lat,lon",
    )

    parser.add_argument(
        "--buffer",
        type=float,
        default=0.01,
        help="Area mode only: bbox buffer in degrees around --center",
    )

    parser.add_argument(
        "--building-file",
        help="Building mode only: path to target building JSON"
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON file path",
    )

    args = parser.parse_args()

    token = os.getenv("MAPILLARY_TOKEN")
    if not token:
        raise ValueError("MAPILLARY_TOKEN environment variable is not set.")

    if args.mode == "area":
        if not args.center:
            parser.error("--center is required for --mode area")
        result = run_area_mode(
            center=parse_coord(args.center),
            buffer_deg=args.buffer,
            token=token,
        )
    if args.mode == "building":
        if not args.building_file:
            parser.error("--building-file is required for building mode")

        result = run_building_mode(
            building_file=args.building_file,
            token=token,
    )

    output_json = json.dumps(result, indent=2)

    if args.output:
        output_file = args.output
    else:
        if args.mode == "area":
            output_file = "data/area_fetch.json"
        else:
            output_file = "data/building_fetch.json"
   
    with open(output_file, "w") as f:
        f.write(output_json)
    print(f"Saved {output_file}")



if __name__ == "__main__":
    main()