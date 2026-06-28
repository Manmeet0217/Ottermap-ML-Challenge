

import json
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.features import rasterize

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GEOREF_DIR = PROJECT_ROOT / "data" / "processed" / "georeferenced"
PSEUDO_DIR = PROJECT_ROOT / "data" / "processed" / "pseudo_labels"
TILES_DIR = PROJECT_ROOT / "data" / "processed" / "tiles"

PARCEL_ID = 3  

with open(TILES_DIR / "class_legend.json") as f:
    CLASS_LEGEND = json.load(f)

image_path = GEOREF_DIR / f"georef_{PARCEL_ID}.tiff"
label_path = PSEUDO_DIR / f"{PARCEL_ID}_pseudo_labeled.geojson"

src = rasterio.open(image_path)
gdf = gpd.read_file(label_path)
if gdf.crs != src.crs:
    gdf = gdf.to_crs(src.crs)

shapes = [(row.geometry, CLASS_LEGEND[row["pred_class"]]) for _, row in gdf.iterrows()]
mask = rasterize(
    shapes, out_shape=(src.height, src.width), transform=src.transform,
    fill=0, dtype="uint8",
)
image = src.read([1, 2, 3]).transpose(1, 2, 0)
src.close()


labeled_fraction = np.mean(mask != 0)
print(f"Parcel {PARCEL_ID}: {labeled_fraction:.1%} of pixels are labeled (non-background)")
print(f"Total polygons: {len(gdf)}")

fig, axes = plt.subplots(1, 2, figsize=(20, 10))
axes[0].imshow(image)
axes[0].set_title(f"Parcel {PARCEL_ID}: full image")
axes[0].axis("off")

axes[1].imshow(image)
axes[1].imshow(mask, cmap="tab10", vmin=0, vmax=9, alpha=0.5)
axes[1].set_title(f"Parcel {PARCEL_ID}: labels overlaid ({labeled_fraction:.1%} labeled)")
axes[1].axis("off")

plt.tight_layout()
plt.show()