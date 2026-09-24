#!/usr/bin/env python3

import argparse
import json
import math
from pathlib import Path

# Candidate gate: only consider image if nearest boundary point is within this radius
R_GATE_M = 120.0

# Sampling resolution along building boundary (meters)
SAMPLE_STEP_M = 2.0

# Camera forward visibility half-angle.
# A sampled building point is considered "in view" only if its relative angle
# to the camera heading is within +/- FOV_HALF_DEG.
FOV_HALF_DEG = 45.0

# Normalization for score:
# score = min(1, visible_span_deg / SPAN_NORM_DEG)
SPAN_NORM_DEG = 60.0

# Reject if only a tiny fraction of the boundary is in front of the camera
MIN_VISIBLE_FRACTION = 0.20

# Reject if the total visible angular span is too small
MIN_VISIBLE_SPAN_DEG = 8.0

# Acceptance threshold on final score
SCORE_MIN = 0.15


def meters_per_degree(lat0: float):
    lat_rad = math.radians(lat0)
    m_per_deg_lat = 111132.92
    m_per_deg_lon = 111412.84 * math.cos(lat_rad)
    return m_per_deg_lat, m_per_deg_lon


def latlon_to_local_xy(lat: float, lon: float, lat0: float, lon0: float):
    mlat, mlon = meters_per_degree(lat0)
    x = (lon - lon0) * mlon
    y = (lat - lat0) * mlat
    return x, y


def local_xy_to_latlon(x: float, y: float, lat0: float, lon0: float):
    mlat, mlon = meters_per_degree(lat0)
    lat = lat0 + (y / mlat)
    lon = lon0 + (x / mlon)
    return float(lat), float(lon)


def bearing_deg_xy(dx: float, dy: float):
    ang = math.degrees(math.atan2(dx, dy))
    return (ang + 360.0) % 360.0


def ang_diff(a: float, b: float):
    d = (a - b) % 360.0
    if d > 180:
        d = 360 - d
    return abs(d)


def signed_ang_diff_deg(target_deg: float, heading_deg: float):
    return (target_deg - heading_deg + 180.0) % 360.0 - 180.0


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


def sample_poly_boundary(poly_xy, step_m=SAMPLE_STEP_M):
    samples = []

    for (ax, ay), (bx, by) in zip(poly_xy[:-1], poly_xy[1:]):
        seg_len = math.hypot(bx - ax, by - ay)
        if seg_len <= 1e-9:
            continue

        n = max(1, int(math.ceil(seg_len / step_m)))
        for i in range(n + 1):
            t = i / n
            sx = ax + t * (bx - ax)
            sy = ay + t * (by - ay)
            samples.append((sx, sy))

    return samples


def visible_building_span(px, py, heading_deg, poly_xy, step_m=SAMPLE_STEP_M, fov_half_deg=FOV_HALF_DEG):
    samples = sample_poly_boundary(poly_xy, step_m=step_m)

    visible_angles = []
    visible_points = []
    all_angles = []

    for sx, sy in samples:
        dx = sx - px
        dy = sy - py
        d = math.hypot(dx, dy)
        if d <= 1e-9:
            continue

        bear = bearing_deg_xy(dx, dy)
        rel = signed_ang_diff_deg(bear, heading_deg)
        all_angles.append(rel)

        if abs(rel) <= fov_half_deg:
            visible_angles.append(rel)
            visible_points.append((sx, sy, d, bear, rel))

    sample_count = len(all_angles)
    visible_count = len(visible_angles)
    visible_fraction = (visible_count / sample_count) if sample_count > 0 else 0.0

    if visible_count == 0:
        return {
            "visible_span_deg": 0.0,
            "visible_count": 0,
            "sample_count": sample_count,
            "visible_fraction": float(visible_fraction),
            "min_visible_angle_deg": None,
            "max_visible_angle_deg": None,
            "center_visible_angle_deg": None,
            "closest_visible_point_xy_m": None,
            "closest_visible_point_latlon": None,
            "closest_visible_distance_m": None,
        }

    min_a = min(visible_angles)
    max_a = max(visible_angles)
    span = max_a - min_a
    center_angle = 0.5 * (min_a + max_a)

    closest_visible = min(visible_points, key=lambda p: p[2])
    cvx, cvy, cvd, _, _ = closest_visible

    return {
        "visible_span_deg": float(span),
        "visible_count": visible_count,
        "sample_count": sample_count,
        "visible_fraction": float(visible_fraction),
        "min_visible_angle_deg": float(min_a),
        "max_visible_angle_deg": float(max_a),
        "center_visible_angle_deg": float(center_angle),
        "closest_visible_point_xy_m": {"x": float(cvx), "y": float(cvy)},
        "closest_visible_distance_m": float(cvd),
        "closest_visible_point_latlon": None,  # filled later in main()
    }


def get_pose(img):
    coords = (img.get("computed_geometry") or {}).get("coordinates")
    heading = img.get("computed_compass_angle")
    if not coords or heading is None:
        return None
    lon, lat = coords[0], coords[1]
    return float(lat), float(lon), float(heading)


def ensure_closed_latlon(coords):
    if not coords:
        return coords
    if coords[0] != coords[-1]:
        return coords + [coords[0]]
    return coords


def main():
    parser = argparse.ArgumentParser(
        description="Filter Mapillary images for a single target building."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to building_fetch.json",
    )
    parser.add_argument(
        "--accepted",
        default="data/intermediate/accepted_target_building.json",
        help="Output JSON for accepted images",
    )
    parser.add_argument(
        "--rejected",
        default="data/intermediate/rejected_target_building.json",
        help="Output JSON for rejected images",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    out_accepted = Path(args.accepted)
    out_rejected = Path(args.rejected)

    data = json.loads(in_path.read_text(encoding="utf-8"))

    if data.get("mode") != "building":
        raise ValueError("Input JSON must be a building-mode fetch result.")

    target_building = data.get("target_building")
    if not target_building:
        raise ValueError("Missing target_building in input JSON.")

    geometry_lat_lon = target_building.get("geometry_lat_lon")
    if not geometry_lat_lon:
        raise ValueError("Missing target_building.geometry_lat_lon in input JSON.")

    geometry_lat_lon = ensure_closed_latlon(geometry_lat_lon)

    accepted = []
    rejected = []

    for img in data.get("mapillary", []) or []:
        base = {
            "id": img.get("id"),
            "thumb_original_url": img.get("thumb_original_url"),
            "computed_geometry": img.get("computed_geometry"),
            "computed_compass_angle": img.get("computed_compass_angle"),
            "captured_at": img.get("captured_at"),
            "target_building_osm_id": target_building.get("osm_id"),
            "target_building_osm_type": target_building.get("osm_type"),
        }

        pose = get_pose(img)
        if pose is None:
            base["reason"] = "missing_pose_or_heading"
            base["metrics"] = {
                "score": None,
                "d_min_m": None,
                "theta_at_dmin_deg": None,
                "visible_span_deg": None,
                "visible_fraction": None,
                "visible_count": None,
                "sample_count": None,
                "score_threshold": float(SCORE_MIN),
                "candidate_gate_m": float(R_GATE_M),
                "sample_step_m": float(SAMPLE_STEP_M),
                "fov_half_deg": float(FOV_HALF_DEG),
                "span_norm_deg": float(SPAN_NORM_DEG),
                "min_visible_fraction": float(MIN_VISIBLE_FRACTION),
                "min_visible_span_deg": float(MIN_VISIBLE_SPAN_DEG),
            }
            rejected.append(base)
            continue

        cam_lat, cam_lon, heading = pose
        lat0 = cam_lat
        lon0 = cam_lon
        cx, cy = 0.0, 0.0

        poly_xy = [
            latlon_to_local_xy(lat, lon, lat0, lon0)
            for (lat, lon) in geometry_lat_lon
        ]

        if len(poly_xy) < 4 or poly_xy[0] != poly_xy[-1]:
            base["reason"] = "invalid_target_building_geometry"
            rejected.append(base)
            continue

        d_min, qx, qy = point_to_poly_min_dist_m(cx, cy, poly_xy)

        if d_min > R_GATE_M:
            base["reason"] = "outside_candidate_gate"
            base["metrics"] = {
                "score": None,
                "d_min_m": float(d_min),
                "theta_at_dmin_deg": None,
                "visible_span_deg": None,
                "visible_fraction": None,
                "visible_count": None,
                "sample_count": None,
                "score_threshold": float(SCORE_MIN),
                "candidate_gate_m": float(R_GATE_M),
                "sample_step_m": float(SAMPLE_STEP_M),
                "fov_half_deg": float(FOV_HALF_DEG),
                "span_norm_deg": float(SPAN_NORM_DEG),
                "min_visible_fraction": float(MIN_VISIBLE_FRACTION),
                "min_visible_span_deg": float(MIN_VISIBLE_SPAN_DEG),
            }
            rejected.append(base)
            continue

        nearest_lat, nearest_lon = local_xy_to_latlon(qx, qy, lat0, lon0)
        theta_at_dmin = ang_diff(heading, bearing_deg_xy(qx - cx, qy - cy))

        span_info = visible_building_span(
            cx,
            cy,
            heading,
            poly_xy,
            step_m=SAMPLE_STEP_M,
            fov_half_deg=FOV_HALF_DEG,
        )

        visible_span_deg = span_info["visible_span_deg"]
        visible_fraction = span_info["visible_fraction"]

        if span_info["closest_visible_point_xy_m"] is not None:
            cvx = span_info["closest_visible_point_xy_m"]["x"]
            cvy = span_info["closest_visible_point_xy_m"]["y"]
            cv_lat, cv_lon = local_xy_to_latlon(cvx, cvy, lat0, lon0)
            span_info["closest_visible_point_latlon"] = {
                "lat": float(cv_lat),
                "lon": float(cv_lon),
            }

        score = min(1.0, visible_span_deg / SPAN_NORM_DEG)

        result = {
            "target_osm_id": target_building.get("osm_id"),
            "target_osm_type": target_building.get("osm_type"),
            "score": float(score),
            "d_min_m": float(d_min),
            "theta_at_dmin_deg": float(theta_at_dmin),
            "visible_span_deg": float(visible_span_deg),
            "visible_fraction": float(visible_fraction),
            "visible_count": int(span_info["visible_count"]),
            "sample_count": int(span_info["sample_count"]),
            "min_visible_angle_deg": span_info["min_visible_angle_deg"],
            "max_visible_angle_deg": span_info["max_visible_angle_deg"],
            "center_visible_angle_deg": span_info["center_visible_angle_deg"],
            "nearest_point_xy_m": {"x": float(qx), "y": float(qy)},
            "nearest_point_latlon": {"lat": float(nearest_lat), "lon": float(nearest_lon)},
            "closest_visible_point_xy_m": span_info["closest_visible_point_xy_m"],
            "closest_visible_point_latlon": span_info["closest_visible_point_latlon"],
            "closest_visible_distance_m": span_info["closest_visible_distance_m"],
        }

        base["metrics"] = {
            "score": float(score),
            "d_min_m": float(d_min),
            "theta_at_dmin_deg": float(theta_at_dmin),
            "visible_span_deg": float(visible_span_deg),
            "visible_fraction": float(visible_fraction),
            "visible_count": int(span_info["visible_count"]),
            "sample_count": int(span_info["sample_count"]),
            "min_visible_angle_deg": span_info["min_visible_angle_deg"],
            "max_visible_angle_deg": span_info["max_visible_angle_deg"],
            "center_visible_angle_deg": span_info["center_visible_angle_deg"],
            "score_threshold": float(SCORE_MIN),
            "candidate_gate_m": float(R_GATE_M),
            "sample_step_m": float(SAMPLE_STEP_M),
            "fov_half_deg": float(FOV_HALF_DEG),
            "span_norm_deg": float(SPAN_NORM_DEG),
            "min_visible_fraction": float(MIN_VISIBLE_FRACTION),
            "min_visible_span_deg": float(MIN_VISIBLE_SPAN_DEG),
        }

        # Hard reject cases where the building is mostly not in view
        if visible_fraction < MIN_VISIBLE_FRACTION:
            base["reason"] = "building_mostly_not_in_view_fraction"
            base["target_match"] = result
            rejected.append(base)
            continue

        if visible_span_deg < MIN_VISIBLE_SPAN_DEG:
            base["reason"] = "building_mostly_not_in_view_span"
            base["target_match"] = result
            rejected.append(base)
            continue

        if score >= SCORE_MIN:
            base["target_match"] = result
            accepted.append(base)
        else:
            base["reason"] = "target_building_score_below_threshold"
            base["target_match"] = result
            rejected.append(base)

    out_accepted.parent.mkdir(parents=True, exist_ok=True)
    out_rejected.parent.mkdir(parents=True, exist_ok=True)

    out_accepted.write_text(json.dumps(accepted, indent=2), encoding="utf-8")
    out_rejected.write_text(json.dumps(rejected, indent=2), encoding="utf-8")

    print(f"Accepted: {len(accepted)}")
    print(f"Rejected: {len(rejected)}")
    print(f"Wrote: {out_accepted}")
    print(f"Wrote: {out_rejected}")


if __name__ == "__main__":
    main()