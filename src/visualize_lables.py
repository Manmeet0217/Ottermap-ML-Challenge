from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import rasterio
from rasterio.plot import show

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Point at the NEW georeferenced output, not the raw image
IMAGE_PATH = PROJECT_ROOT / "data/processed/georeferenced/georef_2.tiff"
LABEL_PATH = PROJECT_ROOT / "data/raw/labels/GeoJSON/2.geojson"

fig, ax = plt.subplots(figsize=(12, 12))

with rasterio.open(IMAGE_PATH) as src:
    # show() is transform-aware: it plots the image in real-world
    # coordinates (using src.transform), not raw pixel indices.
    show(src, ax=ax)

gdf = gpd.read_file(LABEL_PATH)

# Reproject labels to match the image CRS, just in case they ever differ
if gdf.crs is not None:
    with rasterio.open(IMAGE_PATH) as src:
        if src.crs is not None and gdf.crs != src.crs:
            gdf = gdf.to_crs(src.crs)

gdf.boundary.plot(ax=ax, edgecolor="red", linewidth=1.5)

plt.title("Georeferenced Image + GeoJSON Overlay (Parcel 1)")
plt.show()