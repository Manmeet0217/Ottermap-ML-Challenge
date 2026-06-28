import io
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "data" / "external_test"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://imagery.nationalmap.gov/arcgis/rest/services/USGSNAIPImagery/ImageServer/exportImage"


LOCATIONS = [
    ("austin_tx_residential", -97.7431, 30.2672, 0.004),   
    ("denver_co_residential", -104.9903, 39.7392, 0.004), 
    ("miami_fl_residential", -80.1918, 25.7617, 0.004),    
]

IMAGE_SIZE = 2048  


def download_location(name: str, lon: float, lat: float, half_width: float):
    minx = lon - half_width
    maxx = lon + half_width
    miny = lat - half_width
    maxy = lat + half_width

    params = {
        "bbox": f"{minx},{miny},{maxx},{maxy}",
        "bboxSR": 4326,
        "imageSR": 4326,
        "size": f"{IMAGE_SIZE},{IMAGE_SIZE}",
        "format": "tiff",
        "pixelType": "U8",
        "f": "image",
    }

    print(f"Requesting {name} ({lon}, {lat})...")
    resp = requests.get(BASE_URL, params=params, timeout=60)
    resp.raise_for_status()

    content_type = resp.headers.get("Content-Type", "")
    if "image" not in content_type and "tiff" not in content_type:
        print(f"  [!] Unexpected response content-type: {content_type}")
        print(f"  First 300 bytes: {resp.content[:300]}")
        return None

    out_path = OUT_DIR / f"{name}.tif"
    with open(out_path, "wb") as f:
        f.write(resp.content)

    size_kb = out_path.stat().st_size / 1024
    print(f"  Saved -> {out_path} ({size_kb:.0f} KB)")
    return out_path


if __name__ == "__main__":
    print("Downloading external test imagery from USGS NAIP (public domain)...\n")
    saved = []
    for name, lon, lat, half_width in LOCATIONS:
        path = download_location(name, lon, lat, half_width)
        if path:
            saved.append(path)
        print()

    print(f"Done. Saved {len(saved)}/{len(LOCATIONS)} images to {OUT_DIR}")
    if saved:
        print("\nVerify with rasterio before trusting these for inference:")
        print("  python -c \"import rasterio; src = rasterio.open('PATH'); "
              "print(src.crs, src.transform, src.shape)\"")