#!/usr/bin/env python3

import argparse
import json
import math
from pathlib import Path


# Candidate gate: only score buildings whose nearest boundary point is within this radius
R_GATE_M = 120.0

# Sampling resolution along building boundary for theta_min (meters)
SAMPLE_STEP_M = 2.0  # smaller = more accurate, slower

# Scoring decay params
T0_DEG = 40.0   # smaller => penalize angle more strongly
D0_M = 30.0     # smaller => penalize distance more strongly
D1_M = 25.0     # smaller => near-bonus drops faster

# Score weights (should sum to 1.0)
W_MAIN = 0.85   # main term weight (angle * distance)
W_NEAR = 0.15   # near-anywhere bonus term

# Acceptance threshold
SCORE_MIN = 0.08  

# Some buildings are "visibly in front" but the best-angle boundary point can land far away on the footprint.
# This caps the effective distance used in the main distance penalty.
USE_DISTANCE_CAP = True
DISTANCE_CAP_EXTRA_M = 20.0  # d_eff = min(d_at_theta_min, d_min + DISTANCE_CAP_EXTRA_M)

# How many candidates to keep per image for debugging
TOPK = 5

# Reject buildings entirely outside a conservative camera-forward angular cone.
# The old proximity bonus could otherwise admit backward-facing photos.
MAX_VIEW_ANGLE_DEG = 60.0


def meters_per_degree(lat0: float):
    lat_rad = math.radians(lat0)
    m_per_deg_lat = 111132.92
    m_per_deg_lon = 111412.84 * math.cos(lat_rad)
    return m_per_deg_lat, m_per_deg_lon

def latlon_to_local_xy(lat: float, lon: float, lat0: float, lon0: float):
    mlat, mlon = meters_per_degree(lat0)
    x = (lon - lon0) * mlon  # east
    y = (lat - lat0) * mlat  # north
    return x, y

def local_xy_to_latlon(x: float, y: float, lat0: float, lon0: float):
    mlat, mlon = meters_per_degree(lat0)
    lat = lat0 + (y / mlat)
    lon = lon0 + (x / mlon)
    return float(lat), float(lon)


def bearing_deg_xy(dx: float, dy: float):
    # dx east, dy north -> bearing 0=N, 90=E
    ang = math.degrees(math.atan2(dx, dy))
    return (ang + 360.0) % 360.0

def ang_diff(a: float, b: float):
    d = (a - b) % 360.0
    if d > 180:
        d = 360 - d
    return abs(d)


# ----------------------------
# Distance from point to polygon boundary (min over segments)
# Returns: (min_dist, nearest_x, nearest_y)
# poly_xy should be closed (first==last)
# ----------------------------
def point_to_poly_min_dist_m(px, py, poly_xy):
    def seg_dist(px, py, ax, ay, bx, by):
        vx, vy = bx - ax, by - ay
        wx, wy = px - ax, py - ay
        c1 = vx * wx + vy * wy
        if c1 <= 0:
            return math.hypot(px - ax, py - ay), ax, ay
        c2 = vx * vx + vy * vy
        if c2 <= c1:
            return math.hypot(px - bx, py - by), bx, by
        t = c1 / c2
        qx, qy = ax + t * vx, ay + t * vy
        return math.hypot(px - qx, py - qy), qx, qy

    best_d, best_x, best_y = 1e18, None, None
    for (ax, ay), (bx, by) in zip(poly_xy[:-1], poly_xy[1:]):
        d, qx, qy = seg_dist(px, py, ax, ay, bx, by)
        if d < best_d:
            best_d, best_x, best_y = d, qx, qy
    return best_d, best_x, best_y


# ----------------------------
# Compute theta_min + d_at_theta_min by sampling points along polygon boundary
# poly_xy should be closed (first==last)
# Returns: (theta_min_deg, d_at_theta_min_m, best_point_xy)
# ----------------------------
def theta_min_on_boundary(px, py, heading_deg, poly_xy, step_m=SAMPLE_STEP_M):
    best_theta = 1e18
    best_d = None
    best_pt = None

    for (ax, ay), (bx, by) in zip(poly_xy[:-1], poly_xy[1:]):
        seg_len = math.hypot(bx - ax, by - ay)
        if seg_len <= 1e-9:
            continue

        n = max(1, int(math.ceil(seg_len / step_m)))
        for i in range(n + 1):
            t = i / n
            sx = ax + t * (bx - ax)
            sy = ay + t * (by - ay)

            dx = sx - px
            dy = sy - py
            d = math.hypot(dx, dy)
            if d <= 1e-9:
                theta = 0.0
            else:
                bear = bearing_deg_xy(dx, dy)
                theta = ang_diff(heading_deg, bear)

            if theta < best_theta:
                best_theta = theta
                best_d = d
                best_pt = (sx, sy)

    if best_d is None:
        return 180.0, 1e18, None

    return best_theta, best_d, best_pt


# ----------------------------
# Extract OSM building polygons (ways + simple relations)
# Returns list of {"osm_id": "...", "coords": [(lat,lon),...closed...]}
# ----------------------------
def extract_osm_buildings(osm_raw):
    elements = (osm_raw or {}).get("elements", [])
    nodes = {el["id"]: (el["lat"], el["lon"]) for el in elements if el.get("type") == "node"}
    ways = {el["id"]: el for el in elements if el.get("type") == "way"}
    rels = [el for el in elements if el.get("type") == "relation"]

    buildings = []

    def way_coords(way_el):
        coords = []
        for nid in way_el.get("nodes") or []:
            if nid in nodes:
                coords.append(nodes[nid])  # (lat, lon)
        if len(coords) >= 3 and coords[0] != coords[-1]:
            coords.append(coords[0])
        return coords

    # building-tagged ways
    for wid, w in ways.items():
        tags = w.get("tags") or {}
        if "building" not in tags:
            continue
        coords = way_coords(w)
        if len(coords) >= 4:
            buildings.append({"osm_id": f"way/{wid}", "coords": coords})

    # very simple building relations (take first usable outer ring)
    for r in rels:
        tags = r.get("tags") or {}
        if "building" not in tags:
            continue
        members = r.get("members") or []
        outer_way_ids = [m["ref"] for m in members if m.get("type") == "way" and m.get("role") == "outer"]
        for owid in outer_way_ids:
            w = ways.get(owid)
            if not w:
                continue
            coords = way_coords(w)
            if len(coords) >= 4:
                buildings.append({"osm_id": f"relation/{r.get('id')}", "coords": coords})
                break

    return buildings


# ----------------------------
# Mapillary pose
# ----------------------------
def get_pose(img):
    coords = (img.get("computed_geometry") or {}).get("coordinates")
    heading = img.get("computed_compass_angle")
    if not coords or heading is None:
        return None
    lon, lat = coords[0], coords[1]
    return float(lat), float(lon), float(heading)


def filter_area(data: dict):
    """Preserve the existing area geometric scoring; consume post-veg area JSON."""
    if data.get("mode") != "area" or not isinstance(data.get("mapillary"), list):
        raise ValueError("Expected area-mode JSON with a mapillary image list")
    accepted, rejected = [], []
    buildings = extract_osm_buildings(data.get("osm_raw"))

    for img in data["mapillary"]:
        base = dict(img)  # preserve the full Mapillary record and earlier visual scores
        pose = get_pose(img)
        if pose is None:
            base["reason"] = "missing_pose_or_heading"
            base["metrics"] = {"score": None, "best_score": None,
                               "best_osm_id": None}
            rejected.append(base)
            continue

        cam_lat, cam_lon, heading = pose
        cx, cy = 0.0, 0.0
        buildings_xy = []
        for b in buildings:
            coords_xy = [latlon_to_local_xy(lat, lon, cam_lat, cam_lon)
                         for lat, lon in b["coords"]]
            if len(coords_xy) >= 4 and coords_xy[0] == coords_xy[-1]:
                buildings_xy.append((b["osm_id"], coords_xy))

        best, best_score, candidates = None, -1.0, []
        for osm_id, poly in buildings_xy:
            d_min, qx, qy = point_to_poly_min_dist_m(cx, cy, poly)
            if d_min > R_GATE_M:
                continue
            nearest_lat, nearest_lon = local_xy_to_latlon(qx, qy, cam_lat, cam_lon)
            theta_at_dmin = ang_diff(heading, bearing_deg_xy(qx - cx, qy - cy))
            theta_min, d_at_theta_min, best_pt = theta_min_on_boundary(
                cx, cy, heading, poly, step_m=SAMPLE_STEP_M)
            if theta_min > MAX_VIEW_ANGLE_DEG:
                continue  # proximity alone is not evidence the camera sees a building
            d_eff = (min(d_at_theta_min, d_min + DISTANCE_CAP_EXTRA_M)
                     if USE_DISTANCE_CAP else d_at_theta_min)
            g_theta = math.exp(- (theta_min / T0_DEG) ** 2)
            g_dist = math.exp(- d_eff / D0_M)
            g_near = math.exp(- d_min / D1_M)
            score = W_MAIN * g_theta * g_dist + W_NEAR * g_near
            candidate = {
                "osm_id": osm_id, "score": float(score),
                "d_min_m": float(d_min),
                "theta_at_dmin_deg": float(theta_at_dmin),
                "theta_min_deg": float(theta_min),
                "d_at_theta_min_m": float(d_at_theta_min),
                "d_eff_m": float(d_eff),
                "nearest_point_xy_m": {"x": float(qx), "y": float(qy)},
                "nearest_point_latlon": {"lat": float(nearest_lat), "lon": float(nearest_lon)},
                "best_angle_point_xy_m": {"x": float(best_pt[0]), "y": float(best_pt[1])} if best_pt else None,
            }
            candidates.append(candidate)
            if score > best_score:
                best_score, best = score, candidate
        candidates.sort(key=lambda c: c["score"], reverse=True)
        base["top_candidates"] = candidates[:TOPK]
        base["metrics"] = {
            "score": float(best_score) if best else None,  # visualization compatibility
            "best_score": float(best_score) if best else None,
            "best_osm_id": best["osm_id"] if best else None,
            "d_min_m": best["d_min_m"] if best else None,
            "theta_at_dmin_deg": best["theta_at_dmin_deg"] if best else None,
            "theta_min_deg": best["theta_min_deg"] if best else None,
            "d_at_theta_min_m": best["d_at_theta_min_m"] if best else None,
            "d_eff_m": best["d_eff_m"] if best else None,
            "score_threshold": SCORE_MIN,
            "candidate_gate_m": R_GATE_M,
            "max_view_angle_deg": MAX_VIEW_ANGLE_DEG,
            "sample_step_m": SAMPLE_STEP_M,
            "use_distance_cap": USE_DISTANCE_CAP,
            "distance_cap_extra_m": DISTANCE_CAP_EXTRA_M,
        }
        if best is not None and best_score >= SCORE_MIN:
            base["best_match"] = best
            accepted.append(base)
        else:
            base["reason"] = "no_building_scoring_above_threshold"
            base["best_candidate"] = best
            rejected.append(base)
    return accepted, rejected


def main():
    parser = argparse.ArgumentParser(description="Geometric area filter after visual filtering")
    parser.add_argument("--input", required=True, help="Output JSON of first-stage veg.py")
    parser.add_argument("--accepted", required=True, help="Geometrically accepted image list JSON")
    parser.add_argument("--rejected", required=True, help="Geometrically rejected image list JSON")
    args = parser.parse_args()
    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    accepted, rejected = filter_area(data)
    for filename, items in ((args.accepted, accepted), (args.rejected, rejected)):
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(items, indent=2), encoding="utf-8")
    print(f"Geometric stage: {len(accepted)} accepted, {len(rejected)} rejected")
    print(f"Accepted: {args.accepted}")
    print(f"Rejected: {args.rejected}")


if __name__ == "__main__":
    main()
