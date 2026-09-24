# SPECIFICALLY ONLY FOR MODE = BUILDING 
# get the associated osm id of the coordinate input for a single building
import argparse
import json
import os
import requests
from shapely.geometry import Point, Polygon
import time

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

#build a small bounding box around the coordinate
#query Overpass for buildings in that box
#reconstruct building polygons from OSM nodes
#choose the building containing the coordinate (or nearest)

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
[out:json][timeout:60];
(
  way["building"]({s},{w},{n},{e});
);
out geom;
"""

def fetch_osm(bbox, max_retries=3):
    for attempt in range(max_retries):
        try:
            r = requests.post(
                OVERPASS_URL,
                data={"data": overpass_query_buildings(bbox)},
                timeout=120,
            )
            r.raise_for_status()
            return r.json()

        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt+1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)

    raise Exception("Overpass failed after retries")



def extract_buildings(osm_raw):

    buildings = []

    for el in osm_raw["elements"]:
        if el["type"] == "way" and "tags" in el and "building" in el["tags"]:

            coords = [
                (pt["lon"], pt["lat"])
                for pt in el.get("geometry", [])
            ]

            if len(coords) >= 3:
                buildings.append(
                    {
                        "osm_type": "way",
                        "osm_id": el["id"],
                        "tags": el.get("tags", {}),
                        "coords": coords,
                    }
                )

    return buildings

def find_target_building(buildings, lat, lon):

    point = Point(lon, lat)

    containing = []

    for b in buildings:

        poly = Polygon(b["coords"])

        if poly.is_valid and poly.contains(point):
            containing.append((poly.area, b, poly))

    if containing:
        containing.sort(key=lambda x: x[0])
        return containing[0][1], containing[0][2], "contains_point"

    nearest = None
    nearest_poly = None
    nearest_dist = float("inf")

    for b in buildings:

        poly = Polygon(b["coords"])

        if not poly.is_valid:
            continue

        dist = point.distance(poly.centroid)

        if dist < nearest_dist:
            nearest_dist = dist
            nearest = b
            nearest_poly = poly

    return nearest, nearest_poly, "nearest_centroid"


def main():

    parser = argparse.ArgumentParser(
        description="Resolve coordinate to OSM building footprint"
    )

    parser.add_argument(
        "--coord",
        required=True,
        help="Coordinate as lat,lon",
    )

    parser.add_argument(
        "--buffer",
        type=float,
        default=0.0005,
        help="Search buffer in degrees",
    )

    parser.add_argument(
        "--output",
        default="data/target_building.json",
        help="Output JSON file",
    )

    args = parser.parse_args()

    lat, lon = parse_coord(args.coord)

    bbox = bbox_from_center(lat, lon, args.buffer)

    osm_raw = fetch_osm(bbox)

    buildings = extract_buildings(osm_raw)

    if not buildings:
        raise ValueError("No buildings found near coordinate")

    building, poly, method = find_target_building(buildings, lat, lon)

    centroid = poly.centroid

    result = {
        "input_coordinate": {
            "lat": lat,
            "lon": lon,
        },
        "search_buffer_deg": args.buffer,
        "bbox_south_west_north_east": bbox,
        "target_building": {
            "osm_type": building["osm_type"],
            "osm_id": building["osm_id"],
            "tags": building["tags"],
            "geometry_lat_lon": [[lat, lon] for lon, lat in building["coords"]],
            "centroid": {
                "lat": centroid.y,
                "lon": centroid.x,
            },
        },
        "selection_method": method,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()