from pathlib import Path
import rasterio

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGE_DIR = PROJECT_ROOT / "data/raw/images"

for image_path in sorted(IMAGE_DIR.glob("*.tif*")):
    print("\n" + "=" * 80)
    print(image_path.name)

    with rasterio.open(image_path) as src:
        print("Driver:", src.driver)
        print("CRS:", src.crs)
        print("Transform:", src.transform)
        print("Bounds:", src.bounds)
        print("Tags:", src.tags())
        print("Metadata:", src.meta)
        