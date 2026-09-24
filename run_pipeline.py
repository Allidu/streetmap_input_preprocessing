#!/usr/bin/env python3
"""Run the original building pipeline, or area: fetch -> veg -> geometry."""
import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml


def load_config(config_path: str) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_command(cmd: list[str]) -> None:
    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def build_fetch_command(config: dict) -> list[str]:
    mode = config["pipeline"]["mode"]
    cmd = [sys.executable, "src/get_mapillary.py", "--mode", mode]
    if mode == "area":
        a = config["fetch"]["area"]
        cmd.extend(["--center", str(a["center"]), "--buffer", str(a["buffer"]),
                    "--output", a["output"]])
        return cmd
    if mode == "building":
        b = config["fetch"]["building"]
        cmd.extend(["--building-file", b["target_building_output"],
                    "--output", b["output"]])
        return cmd
    raise ValueError(f"Unsupported pipeline mode: {mode}")


def build_vegetation_command(config: dict) -> list[str]:
    v = config["filter"]["vegetation"]
    a = config["fetch"]["area"]
    cmd = [sys.executable, "src/veg.py", "--input", a["output"],
           "--output", v["output"], "--rejected", v["rejected"],
           "--csv", v["csv"], "--contact-sheet", v["contact_sheet"]]
    for option in ("min_building_frac", "max_building_frac",
                   "max_center_vegetation_frac", "max_edge", "max_sheet_images",
                   "model_id", "device"):
        if option in v:
            cmd.extend(["--" + option.replace("_", "-"), str(v[option])])
    return cmd


def build_filter_command(config: dict) -> list[str]:
    mode = config["pipeline"]["mode"]
    if mode == "building":
        return [sys.executable, "src/filter_target_building.py",
                "--input", config["fetch"]["building"]["output"]]
    if mode == "area":
        a = config["filter"]["area"]
        return [sys.executable, "src/filter_metadata.py",
                "--input", config["filter"]["vegetation"]["output"],
                "--accepted", a["output"], "--rejected", a["rejected"]]
    raise ValueError(f"Unsupported pipeline mode: {mode}")


def main():
    p = argparse.ArgumentParser(description="OSM + Mapillary image filtering pipeline")
    p.add_argument("--config", default="configs/default.yaml")
    args = p.parse_args()
    if not os.getenv("MAPILLARY_TOKEN"):
        raise EnvironmentError("MAPILLARY_TOKEN is not set")
    config = load_config(args.config)
    mode = config["pipeline"]["mode"]
    if mode == "building":
        b = config["fetch"]["building"]
        Path(b["target_building_output"]).parent.mkdir(parents=True, exist_ok=True)
        run_command([sys.executable, "src/coords_to_osm_building.py", "--coord",
                     str(b["building_coord"]), "--output", b["target_building_output"]])
    fetch_cmd = build_fetch_command(config)
    Path(fetch_cmd[-1]).parent.mkdir(parents=True, exist_ok=True)
    run_command(fetch_cmd)
    if mode == "area":
        run_command(build_vegetation_command(config))  # ALWAYS before geometric filter
    run_command(build_filter_command(config))


if __name__ == "__main__":
    main()
