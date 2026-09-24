# Mapillary image filtering

Select street-level photos of buildings. Area mode fetches a neighborhood, drops photos with too little building or too much vegetation, then keeps photos whose camera points at a nearby OSM building.

## Use the pipeline

From this directory:

```bash
pip install -r requirements.txt
export MAPILLARY_TOKEN=your_token
python run_pipeline.py --config configs/default.yaml
```

Do not put the token in the YAML file.

That command runs three stages, in order:

1. Fetch OSM buildings and Mapillary image records.
2. Vegetation filter. Rejected photos do not continue.
3. Geometric filter, on the vegetation-accepted photos only.

Optional fourth step, after the pipeline finishes. This ranks the geometric accepts and can drop near-duplicate frames:

```bash
python src/select_best_building_views.py --config configs/default.yaml
```

Turn that dedup off with `filter.dedup.enabled: false` in the config, or pass `--no-dedup`.

The first segmentation run downloads the Mask2Former weights. Inference needs a network connection. On CPU it is slow for large areas. `device: -1` is CPU; `0` is the first CUDA GPU.

## Change the config

All of these settings are in `configs/default.yaml`.

`pipeline.mode` is `area` or `building`. Area mode searches a box and runs vegetation filtering before geometry. Building mode looks up one OSM building and uses the original building filter, with no vegetation stage.

`fetch.area.center` is `"lat,lon"`. `fetch.area.buffer` is the box half-width in degrees, not meters. `0.0015` is a few city blocks. Larger values pull in more buildings and more photos. `fetch.area.output` is where the raw JSON is saved.

`fetch.building.building_coord` is the point used to find one target building in building mode.

`filter.vegetation.min_building_frac` rejects a photo when too little of the frame is building. Default `0.08`.

`filter.vegetation.max_building_frac` rejects a photo that is almost entirely building. Default `0.93`.

`filter.vegetation.max_center_vegetation_frac` rejects a photo when vegetation fills the center 70% of the frame. Default `0.55`. This is vegetation occupancy in that crop, not a measurement of how much façade is hidden by trees.

`filter.vegetation.model_id` is the segmentation model. `device` chooses CPU or GPU. `max_edge` is the longest image side sent to the model. `max_sheet_images` caps the contact sheet.

`filter.area.output` and `filter.area.rejected` are the geometric accept and reject files. The score cutoff, 60° forward-view limit, and distance weights are constants in `src/filter_metadata.py`, not in the YAML.

`filter.dedup` is used only by `src/select_best_building_views.py`. `enabled` turns near-duplicate removal on or off. Two photos are duplicates when they are within `distance_m` meters and `heading_deg` degrees. Each cluster keeps the highest-scoring photo. There is no top-N cutoff. The per-building cap of 5 is a constant in that script.

## What each file does

`run_pipeline.py` reads the config and runs fetch, then vegetation, then geometry.

`src/get_mapillary.py` downloads OSM buildings and Mapillary metadata, including image URLs. It does not save an image folder.

`src/veg.py` downloads each photo, segments it, and writes accepted records, rejected records, a CSV, and a contact sheet. Building mask is green. Vegetation mask is orange.

`src/filter_metadata.py` scores each vegetation-accepted photo against nearby OSM footprints using camera position and compass heading.

`src/select_best_building_views.py` ranks geometric accepts by building visibility and low vegetation, then optionally removes near-duplicates.

`src/coords_to_osm_building.py` turns one coordinate into a target building. Used by building mode.

`src/filter_target_building.py` is the original single-building filter. Building mode only.

`src/visualize_filter.py` writes an HTML gallery for a filtered image list.

`src/visualize_eval.py` builds the Thompson Street evaluation gallery, map, and comparison sheet from a finished run.

`src/visualize2.py` writes an HTML gallery for a raw fetch file.

`src/organize_directions.py` groups images by compass direction.

`src/places.py` is a Places365 scene classifier. The pipeline does not run it.

`tests/test_area_pipeline.py` checks that vegetation filtering runs before geometry and that metadata is kept. It does not call Mapillary or load the model.
