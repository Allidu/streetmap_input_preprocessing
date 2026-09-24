# Area-mode vegetation-first Mapillary pipeline

The supplied project keeps the existing code flow: `python run_pipeline.py --config configs/default.yaml`. Run from this directory after setting `MAPILLARY_TOKEN` in your environment. `configs/default.yaml` now defaults to `mode: area`. Your original `get_mapillary.py`, `coords_to_osm_building.py`, and `filter_target_building.py` are included without changes to their algorithms. **Only area mode has a vegetation-first integration; building mode retains its original filtering flow.** Places365 is not run automatically.

Install dependencies with `pip install -r requirements.txt`. The first real segmentation run downloads the Mask2Former model weights; processing all Mapillary photos on CPU can be slow and requires network connectivity. `get_mapillary.py` uses Overpass and Mapillary endpoints, so set `MAPILLARY_TOKEN` before running. Never put the token in the YAML file or share it.

## Area-mode stages and outputs

1. `src/get_mapillary.py`: fetches OSM buildings and Mapillary image records (metadata and URL) into `data/raw/area_fetch.json`; this stage does **not** download and retain an image directory.
2. `src/veg.py`: retrieves pixels from `thumb_original_url` (or from a record's existing `local_path` / `image_path`, if available), performs segmentation, and writes `data/intermediate/area_vegetation_accepted.json`. This file preserves the original fetched JSON object, including `osm_raw`, and changes only its `mapillary` list to the visually accepted records. Every record retains its original Mapillary fields plus `visual_filter` metrics. Separate rejected records and a per-image CSV are also written; the contact sheet includes up to 80 successfully evaluated images.
3. `src/filter_metadata.py`: reads only those visually accepted images, computes the original per-image geometric scores against nearby OSM footprints, and writes `data/filtered/area_filtered.json` as a **list** of accepted image records, plus `data/filtered/area_geometric_rejected.json`. Each output record retains its `visual_filter` metrics and other Mapillary metadata.

You can inspect the final accepted list with the existing visualization script, e.g. `python src/visualize_filter.py --input data/filtered/area_filtered.json --output data/outputs/area_gallery.html`. The separate `data/intermediate/area_vegetation_rejected.json` and `data/filtered/area_geometric_rejected.json` tell you exactly which stage removed each photo.

## Important interpretation and limits

The original `vegetation_mask & building_mask` calculation **cannot measure occlusion** with mutually exclusive panoptic labels: it is normally zero. Instead, `veg.py` reports (a) overall building fraction, (b) overall vegetation fraction, and (c) vegetation fraction in the central 70% × 70% crop. By default it rejects building coverage below 8%, building coverage above 93%, or center vegetation occupancy above 55%. The last threshold is a **conservative image-level vegetation dominance heuristic**, not a percentage of a façade physically occluded by trees. Tune thresholds after visually checking your sample region; images showing unobstructed but small buildings can be rejected by the 8% rule.

Segmentation/download errors are marked `image_or_segmentation_error` and are not passed to geometric filtering. If every input image fails evaluation, the stage fails rather than treating the run as successful. The contact sheet is intentionally capped to avoid an enormous image. The geometric filter retains the preexisting distance-and-heading scoring but adds a conservative 60° forward-view gate: previously, the proximity bonus could admit a nearby building behind a camera. This gate is only a rough horizontal field-of-view estimate, not a true visibility check. It does not estimate inter-image feature overlap, true façade visibility, or reconstructability and does **not** yet implement redundancy removal or a fixed final photo count.

For a smaller test area, reduce `fetch.area.buffer` in YAML and inspect the first-stage contact sheet and CSV before running full-scale inference.
