from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import rasterio
from rasterio.plot import show
from rasterio.windows import from_bounds as window_from_bounds

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GEOREF_DIR = PROJECT_ROOT / "data" / "processed" / "georeferenced"
PSEUDO_DIR = PROJECT_ROOT / "data" / "processed" / "pseudo_labels"


PARCEL_ID = 1
FEATURE_INDICES = [16, 13, 34, 65, 49]  

image_path = GEOREF_DIR / f"georef_{PARCEL_ID}.tiff"
label_path = PSEUDO_DIR / f"{PARCEL_ID}_pseudo_labeled.geojson"

gdf = gpd.read_file(label_path)
src = rasterio.open(image_path)

n = len(FEATURE_INDICES)
fig, axes = plt.subplots(1, n, figsize=(6 * n, 6))
if n == 1:
    axes = [axes]

for ax, idx in zip(axes, FEATURE_INDICES):
    row = gdf.iloc[idx]
    geom = row.geometry
    minx, miny, maxx, maxy = geom.bounds

    pad_x = (maxx - minx) * 1.5 + 1e-6
    pad_y = (maxy - miny) * 1.5 + 1e-6

    window = window_from_bounds(
        minx - pad_x, miny - pad_y, maxx + pad_x, maxy + pad_y,
        transform=src.transform
    )
    patch = src.read([1, 2, 3], window=window)
    patch_transform = src.window_transform(window)

    ax.imshow(patch.transpose(1, 2, 0),
              extent=[minx - pad_x, maxx + pad_x, miny - pad_y, maxy + pad_y])

    gpd.GeoSeries([geom]).boundary.plot(ax=ax, edgecolor="red", linewidth=2)

    ax.set_title(f"idx={idx}\n{row['pred_class']} ({row['confidence']:.2f})")
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()
src.close()