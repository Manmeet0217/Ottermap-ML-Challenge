import rasterio
from rasterio.transform import from_bounds
import geopandas as gpd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
IMAGES_DIR = DATA_DIR / "images"
LABELS_DIR = DATA_DIR / "labels" / "GeoJSON"
OUT_DIR = PROJECT_ROOT / "data" / "processed" / "georeferenced"
OUT_DIR.mkdir(parents=True, exist_ok=True)


IMAGE_LABEL_PAIRS = [
    ("1.tiff", "1.geojson"),   
    ("2.tiff", "2.geojson"),
    ("3.tiff", "3.geojson"),
]


def diagnose(image_path: Path, label_path: Path):
    print(f"\n--- {image_path.name} <-> {label_path.name} ---")

    with rasterio.open(image_path) as src:
        print(f"Image size (w,h): {src.width} x {src.height}")
        print(f"Image CRS: {src.crs}")
        print(f"Image transform: {src.transform}")

    gdf = gpd.read_file(label_path)
    print(f"Label CRS: {gdf.crs}")
    minx, miny, maxx, maxy = gdf.total_bounds
    print(f"Label bounds (lon/lat): minx={minx:.6f}, miny={miny:.6f}, "
          f"maxx={maxx:.6f}, maxy={maxy:.6f}")


def reconstruct_geotransform(image_path: Path, label_path: Path, out_path: Path):
    gdf = gpd.read_file(label_path)
    minx, miny, maxx, maxy = gdf.total_bounds

    with rasterio.open(image_path) as src:
        width, height = src.width, src.height
        data = src.read()
        profile = src.profile.copy()

    
    new_transform = from_bounds(minx, miny, maxx, maxy, width, height)

    profile.update({
        "crs": "EPSG:4326",
        "transform": new_transform,
    })

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(data)

    print(f"Wrote georeferenced image -> {out_path}")
    print(f"New transform: {new_transform}")


if __name__ == "__main__":
    print("=== STEP 1: Diagnose current state ===")
    for img_name, label_name in IMAGE_LABEL_PAIRS:
        img_path = IMAGES_DIR / img_name
        label_path = LABELS_DIR / label_name
        if img_path.exists() and label_path.exists():
            diagnose(img_path, label_path)
        else:
            print(f"\n[!] Missing file(s) for pair: {img_name}, {label_name}")
            print(f"    Check actual filenames in {IMAGES_DIR} and {LABELS_DIR}")

    print("\n=== STEP 2: Reconstruct geotransform from label bounds ===")
    for img_name, label_name in IMAGE_LABEL_PAIRS:
        img_path = IMAGES_DIR / img_name
        label_path = LABELS_DIR / label_name
        if img_path.exists() and label_path.exists():
            out_path = OUT_DIR / f"georef_{img_name}"
            reconstruct_geotransform(img_path, label_path, out_path)