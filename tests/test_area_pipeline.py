"""No external APIs or pretrained model needed for these tests."""
import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


veg = load("area_veg", ROOT / "src/veg.py")
geo = load("area_geo", ROOT / "src/filter_metadata.py")
runner = load("area_runner", ROOT / "run_pipeline.py")


def fixtures(tmp_path):
    # Closed OSM building to the north of the camera. This is synthetic data.
    coords = [(42.0001, -83.00004), (42.0001, -82.99996),
              (42.00015, -82.99996), (42.00015, -83.00004)]
    elements = [{"type": "node", "id": i, "lat": lat, "lon": lon}
                for i, (lat, lon) in enumerate(coords, start=1)]
    elements.append({"type": "way", "id": 7, "tags": {"building": "yes"},
                     "nodes": [1, 2, 3, 4, 1]})
    photos = []
    for name, rgb, heading in (("good", (1, 0, 0), 0),
                               ("tree", (2, 0, 0), 0),
                               ("wrong_heading", (3, 0, 0), 180)):
        image_file = tmp_path / (name + ".png")
        Image.new("RGB", (100, 100), rgb).save(image_file)
        photos.append({"id": name, "local_path": str(image_file),
                       "thumb_original_url": "https://example.invalid/" + name,
                       "camera_parameters": [1, 2, 3],
                       "computed_geometry": {"coordinates": [-83.0, 42.0]},
                       "computed_compass_angle": heading})
    return {"mode": "area", "osm_raw": {"elements": elements},
            "mapillary": photos, "center": {"lat": 42, "lon": -83}}


def segmenter(image):
    marker = int(np.array(image)[0, 0, 0])
    building = np.zeros((image.height, image.width), dtype=np.uint8)
    veg_mask = np.zeros_like(building)
    if marker == 2:
        building[:, :20] = 1
        veg_mask[:, 20:] = 1
    else:
        building[:, :60] = 1
    return [{"label": "building", "mask": building},
            {"label": "vegetation", "mask": veg_mask}]


def test_first_filter_then_geometry_and_metadata_preserved(tmp_path):
    raw = fixtures(tmp_path)
    class NoNetwork:
        def get(self, *args, **kwargs):
            raise AssertionError("Local path should be used instead of URL")
    filtered, rejected, rows, thumbs = veg.filter_area(
        raw, segmenter, NoNetwork(), min_building_frac=0.08,
        max_building_frac=0.93, max_center_vegetation_frac=0.55,
        max_edge=1024)
    assert [x["id"] for x in filtered["mapillary"]] == ["good", "wrong_heading"]
    assert [x["id"] for x in rejected] == ["tree"]
    assert rejected[0]["visual_filter"]["reason"] == "vegetation_dominates_center"
    assert raw["mapillary"][1]["id"] == "tree"  # original fetched data unchanged
    assert filtered["osm_raw"] == raw["osm_raw"]
    assert filtered["mapillary"][0]["camera_parameters"] == [1, 2, 3]
    assert len(rows) == len(thumbs) == 3
    accepted_geo, rejected_geo = geo.filter_area(filtered)
    assert [x["id"] for x in accepted_geo] == ["good"]
    assert [x["id"] for x in rejected_geo] == ["wrong_heading"]
    assert accepted_geo[0]["visual_filter"]["building_frac"] == 0.6
    assert accepted_geo[0]["metrics"]["score"] >= geo.SCORE_MIN
    assert accepted_geo[0]["best_match"]["osm_id"] == "way/7"


def test_masks_are_not_interpreted_as_occlusion():
    building = np.zeros((100, 100), bool)
    building[:, :40] = True
    veg_mask = ~building
    m = veg.calculate_metrics(building, veg_mask)
    assert m["building_frac"] == 0.4
    assert m["vegetation_frac"] == 0.6
    assert not (building & veg_mask).any()
    assert m["center_vegetation_frac"] > 0.55


def test_empty_and_error_handling(tmp_path):
    data = fixtures(tmp_path)
    data["mapillary"][0]["local_path"] = "missing.png"
    data["mapillary"][0]["thumb_original_url"] = None
    class NoNetwork:
        def get(self, *args, **kwargs):
            raise AssertionError("should not download this image")
    out, reject, rows, _ = veg.filter_area(
        data, segmenter, NoNetwork(), min_building_frac=0.08,
        max_building_frac=0.93, max_center_vegetation_frac=0.55,
        max_edge=1024, max_sheet_images=0)
    assert len(out["mapillary"]) == 1
    assert [r["id"] for r in reject] == ["good", "tree"]
    assert reject[0]["visual_filter"]["reason"] == "image_or_segmentation_error"
    assert len(rows) == 3
    empty = {"mode": "area", "mapillary": [], "osm_raw": {"elements": []}}
    assert veg.filter_area(empty, segmenter, NoNetwork(), min_building_frac=0.08,
                           max_building_frac=0.93, max_center_vegetation_frac=0.55,
                           max_edge=1024)[0]["mapillary"] == []
    assert geo.filter_area(empty) == ([], [])


def test_runner_enforces_order_and_paths(monkeypatch, tmp_path):
    config = runner.load_config(ROOT / "configs/default.yaml")
    assert config["pipeline"]["mode"] == "area"
    commands = []
    monkeypatch.setenv("MAPILLARY_TOKEN", "test-only-not-used")
    monkeypatch.setattr(runner, "run_command", lambda command: commands.append(command))
    monkeypatch.setattr(runner.sys, "argv", ["run_pipeline.py", "--config", str(ROOT / "configs/default.yaml")])
    monkeypatch.chdir(tmp_path)
    runner.main()
    assert [cmd[1] for cmd in commands] == ["src/get_mapillary.py", "src/veg.py", "src/filter_metadata.py"]
    assert commands[1][commands[1].index("--input") + 1] == config["fetch"]["area"]["output"]
    assert commands[2][commands[2].index("--input") + 1] == config["filter"]["vegetation"]["output"]
    assert commands[2][commands[2].index("--accepted") + 1] == config["filter"]["area"]["output"]


def test_runner_skips_fetch_when_configured(monkeypatch, tmp_path):
    config = runner.load_config(ROOT / "configs/default.yaml")
    config["fetch"]["skip"] = True
    raw = tmp_path / "area_fetch.json"
    raw.write_text("{}", encoding="utf-8")
    config["fetch"]["area"]["output"] = str(raw)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    commands = []
    monkeypatch.delenv("MAPILLARY_TOKEN", raising=False)
    monkeypatch.setattr(runner, "run_command", lambda command: commands.append(command))
    monkeypatch.setattr(runner.sys, "argv", ["run_pipeline.py", "--config", str(config_path)])
    runner.main()
    assert [cmd[1] for cmd in commands] == ["src/veg.py", "src/filter_metadata.py"]
    assert commands[0][commands[0].index("--input") + 1] == str(raw)


def test_mapillary_url_image_is_loaded_without_prior_local_download():
    import io
    image_bytes = io.BytesIO()
    Image.new("RGB", (8, 6), (7, 8, 9)).save(image_bytes, format="PNG")
    class Response:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def raise_for_status(self):
            pass
        def iter_content(self, chunk_size):
            yield image_bytes.getvalue()
    class Session:
        def get(self, url, **kwargs):
            assert url == "https://example.com/mapillary-photo.png"
            assert kwargs["stream"] is True
            return Response()
    loaded = veg.open_image({"id": "remote", "thumb_original_url":
                            "https://example.com/mapillary-photo.png"}, Session())
    assert loaded.size == (8, 6)
    assert loaded.getpixel((0, 0)) == (7, 8, 9)


def test_cli_stages_write_expected_files(monkeypatch, tmp_path):
    import sys
    import types
    raw = fixtures(tmp_path)
    raw_file = tmp_path / "area_fetch.json"
    raw_file.write_text(json.dumps(raw), encoding="utf-8")
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.pipeline = lambda task, model, device: segmenter
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    first = tmp_path / "intermediate/area_veg_accepted.json"
    visual_rejected = tmp_path / "intermediate/area_veg_rejected.json"
    csv_file = tmp_path / "outputs/scores.csv"
    sheet = tmp_path / "outputs/contact.jpg"
    monkeypatch.setattr(sys, "argv", ["veg.py", "--input", str(raw_file),
                       "--output", str(first), "--rejected", str(visual_rejected),
                       "--csv", str(csv_file), "--contact-sheet", str(sheet)])
    veg.main()
    assert first.exists() and visual_rejected.exists() and csv_file.exists() and sheet.exists()
    assert len(json.loads(first.read_text())["mapillary"]) == 2
    geometric_accepted = tmp_path / "filtered/final.json"
    geometric_rejected = tmp_path / "filtered/rejected.json"
    monkeypatch.setattr(sys, "argv", ["filter_metadata.py", "--input", str(first),
                       "--accepted", str(geometric_accepted),
                       "--rejected", str(geometric_rejected)])
    geo.main()
    assert [item["id"] for item in json.loads(geometric_accepted.read_text())] == ["good"]
    assert [item["id"] for item in json.loads(geometric_rejected.read_text())] == ["wrong_heading"]
