from pathlib import Path

import geopandas as gpd
import rasterio

PROJECT_ROOT = Path(__file__).resolve().parent.parent

IMAGE_DIR = PROJECT_ROOT / "data/raw/images"
LABEL_DIR = PROJECT_ROOT / "data/raw/labels/GeoJSON"

print("=" * 70)
print("IMAGE INFORMATION")
print("=" * 70)

for image_path in sorted(IMAGE_DIR.glob("*.tif*")):
    with rasterio.open(image_path) as src:
        print(f"\nImage: {image_path.name}")
        print(f"Width: {src.width}")
        print(f"Height: {src.height}")
        print(f"Bands: {src.count}")
        print(f"CRS: {src.crs}")
        print(f"Bounds: {src.bounds}")
        print(f"Resolution: {src.res}")

print("\n" + "=" * 70)
print("LABEL INFORMATION")
print("=" * 70)

for geojson_path in sorted(LABEL_DIR.glob("*.geojson")):
    gdf = gpd.read_file(geojson_path)

    print(f"\nLabel: {geojson_path.name}")
    print(f"Features: {len(gdf)}")
    print(f"CRS: {gdf.crs}")
    print(f"Columns: {list(gdf.columns)}")