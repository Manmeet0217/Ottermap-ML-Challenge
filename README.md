# Ottermap ML Internship Challenge — Geospatial Feature Detection

A computer vision pipeline that detects geospatial features (buildings, turf,
woody vegetation, parking lots, roads, sidewalks, water) from aerial orthophotos
and outputs GIS-compatible vector data (GeoJSON).

> **Note on large files:** the trained model checkpoint (`weights/best_model.pth`,
> ~93MB) and the full dataset (`data/`, ~190MB — raw imagery, processed tiles,
> external test images) are **not included in this repository** due to file
> size, and are shared separately via Google Drive:
> **[INSERT YOUR GOOGLE DRIVE LINK HERE]**
>
> This repo contains all source code needed to regenerate both from scratch
> (see *Training workflow* below), as well as sample outputs (`outputs/`,
> `results/`) and training metrics (`reports/`) so the pipeline's behavior
> can be reviewed without downloading the full dataset.

## Setup

```bash
git clone <this-repo-url>
cd ottermap-challenge
python3 -m venv .venv
source .venv/bin/activate
pip install torch torchvision transformers accelerate rasterio geopandas \
            shapely fiona opencv-python albumentations matplotlib \
            scikit-learn scikit-image segmentation_models_pytorch scipy requests
```

Tested on macOS (Apple Silicon, MPS backend). Falls back to CPU automatically
if MPS/CUDA are unavailable.

To run inference using the pretrained checkpoint, first download
`best_model.pth` from the Google Drive link above and place it at
`weights/best_model.pth`.

## Project structure

```
├── inference.py              # Official entry point (see Inference below)
├── src/
│   ├── fix_georeferencing.py     # Recovers missing geotransform from label bounds
│   ├── clip_pseudo_label.py      # CLIP zero-shot labeling (source data had no class labels)
│   ├── rasterize_and_tile.py     # Vector labels -> raster masks -> 256x512 tiles
│   ├── dataset.py                # Train/val split + PyTorch Dataset
│   ├── train.py                  # Model training
│   ├── download_external_imagery.py  # Fetches USGS NAIP imagery for generalization test
│   └── generate_sample_overlays.py   # Produces results/ comparison images
├── data/
│   ├── raw/                  # Original Ottermap-provided imagery + labels
│   ├── processed/            # Georeferenced images, pseudo-labels, tiles
│   └── external_test/        # USGS NAIP imagery (different locations, for generalization test)
├── weights/best_model.pth    # Trained model checkpoint
├── results/                  # Sample predictions: training/validation/external
├── outputs/                  # inference.py output (mask, overlay, GeoJSON) per run
└── reports/training_history.json
```

## Training workflow

The training data (3 aerial orthophotos with GeoJSON labels) required two
non-trivial preprocessing fixes before training was possible:

1. **Georeferencing recovery** — the provided TIFFs had no embedded
   CRS/transform. Reconstructed via `src/fix_georeferencing.py`, using each
   image's corresponding label bounding box as a proxy for the image extent.
   Verified visually (polygons align correctly with real building/feature
   outlines).
2. **Missing class labels** — the provided GeoJSON polygons had no class
   attributes (only null `id`/`Area` fields). Used CLIP zero-shot
   classification (`src/clip_pseudo_label.py`) to assign each polygon to one
   of 7 classes, verified via two rounds of visual + statistical spot-checking.

To reproduce from raw data:

```bash
python src/fix_georeferencing.py
python src/clip_pseudo_label.py
python src/rasterize_and_tile.py
python src/train.py
```

`train.py` trains a U-Net (pretrained ResNet34 encoder) with class-weighted
loss, encoder freezing for the first 10 epochs, and early stopping. See
`TECHNICAL_SUMMARY.pdf` for full methodology, training curves, and the
class-imbalance/data-scarcity tradeoffs encountered.

## Inference

Official evaluation command, run from the project root:

```bash
python inference.py --image path/to/your_image.tif
```

Optional arguments:
```bash
python inference.py --image path/to/image.tif --weights weights/best_model.pth --output-dir outputs/
```

This loads the trained checkpoint, tiles the input image, runs the
segmentation model, applies noise-smoothing, and writes three files to
`outputs/`:
- `<name>_mask.png` — raw per-pixel class-ID mask
- `<name>_overlay.png` — color-coded visualization overlaid on the source image
- `<name>.geojson` — vector polygons in the image's CRS, one feature per
  detected region (background excluded), suitable for direct use in QGIS/ArcGIS

**Note:** the input image should ideally have a valid embedded CRS/transform
for the GeoJSON output to have meaningful real-world coordinates. If the CRS
is missing, the script will still run and produce pixel-space output, with a
warning printed.

## Known limitations

See `TECHNICAL_SUMMARY.pdf` for full discussion. Briefly:
- Trained on only 3 source images (264 training tiles after filtering) —
  model performance is data-limited, not architecture-limited.
- Rare classes (road, water, sidewalk, parking_lot) have low instance
  diversity in the source data and are not reliably learned.
- Tiled inference produces visible grid-line seams at tile boundaries
  (independent per-tile classification, no overlap blending).
- Generalization to genuinely different urban morphology (tested on dense
  downtown imagery from 3 US cities) is weak, as expected given the
  suburban/sports-complex training distribution.