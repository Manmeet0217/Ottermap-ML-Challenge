import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import rasterize
import geopandas as gpd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GEOREF_DIR = PROJECT_ROOT / "data" / "processed" / "georeferenced"
PSEUDO_DIR = PROJECT_ROOT / "data" / "processed" / "pseudo_labels"
TILES_DIR = PROJECT_ROOT / "data" / "processed" / "tiles"
TILE_IMAGES_DIR = TILES_DIR / "images"
TILE_MASKS_DIR = TILES_DIR / "masks"

TILE_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
TILE_MASKS_DIR.mkdir(parents=True, exist_ok=True)


TILE_H = 256
TILE_W = 512
STEP_H = TILE_H
STEP_W = TILE_W
MAX_BACKGROUND_FRACTION = 0.95


CLASS_LEGEND = {
    "background": 0,
    "building": 1,
    "turf_grass": 2,
    "woody_vegetation": 3,
    "parking_lot": 4,
    "road": 5,
    "sidewalk": 6,
    "water": 7,
}


def rasterize_parcel(parcel_id: int):
    image_path = GEOREF_DIR / f"georef_{parcel_id}.tiff"
    label_path = PSEUDO_DIR / f"{parcel_id}_pseudo_labeled.geojson"

    src = rasterio.open(image_path)
    gdf = gpd.read_file(label_path)
    if gdf.crs != src.crs:
        gdf = gdf.to_crs(src.crs)

    
    shapes = []
    skipped = 0
    for _, row in gdf.iterrows():
        class_name = row["pred_class"]
        class_id = CLASS_LEGEND.get(class_name)
        if class_id is None:
            skipped += 1
            continue
        shapes.append((row.geometry, class_id))

    if skipped:
        print(f"  [!] skipped {skipped} polygon(s) with unknown class name")

    mask = rasterize(
        shapes,
        out_shape=(src.height, src.width),
        transform=src.transform,
        fill=CLASS_LEGEND["background"],
        dtype="uint8",
    )

    image = src.read([1, 2, 3]).transpose(1, 2, 0)  
    src.close()

    print(f"  Rasterized mask shape: {mask.shape}, unique classes present: {np.unique(mask)}")
    return image, mask


def tile_parcel(parcel_id: int, image: np.ndarray, mask: np.ndarray):
    h, w = mask.shape
    n_saved = 0
    n_discarded = 0

    for row_start in range(0, h - TILE_H + 1, STEP_H):
        for col_start in range(0, w - TILE_W + 1, STEP_W):
            img_tile = image[row_start:row_start + TILE_H, col_start:col_start + TILE_W]
            mask_tile = mask[row_start:row_start + TILE_H, col_start:col_start + TILE_W]

            background_fraction = np.mean(mask_tile == CLASS_LEGEND["background"])
            if background_fraction > MAX_BACKGROUND_FRACTION:
                n_discarded += 1
                continue

            tile_name = f"{parcel_id}_{row_start}_{col_start}.png"
            Image.fromarray(img_tile).save(TILE_IMAGES_DIR / tile_name)
            Image.fromarray(mask_tile, mode="L").save(TILE_MASKS_DIR / tile_name)
            n_saved += 1

    
    print(f"  Saved {n_saved} tiles, discarded {n_discarded} (mostly background)")
    return n_saved, n_discarded


if __name__ == "__main__":
    with open(TILES_DIR / "class_legend.json", "w") as f:
        json.dump(CLASS_LEGEND, f, indent=2)
    print(f"Saved class legend -> {TILES_DIR / 'class_legend.json'}")
    print(f"Classes: {CLASS_LEGEND}\n")

    total_saved = 0
    total_discarded = 0
    for parcel_id in [1, 2, 3]:
        print(f"--- Parcel {parcel_id} ---")
        image, mask = rasterize_parcel(parcel_id)
        saved, discarded = tile_parcel(parcel_id, image, mask)
        total_saved += saved
        total_discarded += discarded
        print()

    print(f"TOTAL: {total_saved} tiles saved, {total_discarded} discarded across all parcels")
    if total_saved < 30:
        print("\n[!] WARNING: very few tiles saved overall. With this little data,")
        print("    consider lowering MAX_BACKGROUND_FRACTION or using overlapping")
        print("    tiles (smaller STEP_H/STEP_W) to get more training examples.")