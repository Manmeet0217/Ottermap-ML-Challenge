import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import segmentation_models_pytorch as smp
import torch
from PIL import Image
from rasterio.features import shapes as rasterio_shapes
from shapely.geometry import shape as shapely_shape

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_WEIGHTS_PATH = PROJECT_ROOT / "weights" / "best_model.pth"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"

TILE_H = 256
TILE_W = 512

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


CLASS_COLORS = {
    "background": (0, 0, 0),
    "building": (230, 25, 75),
    "turf_grass": (60, 180, 75),
    "woody_vegetation": (0, 100, 0),
    "parking_lot": (128, 128, 128),
    "road": (70, 70, 200),
    "sidewalk": (245, 130, 48),
    "water": (0, 200, 255),
}


def load_model(weights_path: Path):
    checkpoint = torch.load(weights_path, map_location=DEVICE, weights_only=False)
    class_legend = checkpoint["class_legend"]
    num_classes = checkpoint["num_classes"]
    encoder_name = checkpoint.get("encoder_name", "resnet34")

    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=None,  
        in_channels=3,
        classes=num_classes,
    ).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Loaded model from {weights_path}")
    print(f"  Trained for {checkpoint.get('epoch', '?')} epochs, "
          f"val mIoU={checkpoint.get('val_mIoU', float('nan')):.4f}")
    print(f"  Classes: {class_legend}")

    return model, class_legend


def run_inference_on_image(model, image: np.ndarray, num_classes: int):
    
    h, w = image.shape[:2]
    full_mask = np.zeros((h, w), dtype=np.uint8)

    with torch.no_grad():
        for row_start in range(0, h, TILE_H):
            for col_start in range(0, w, TILE_W):
                row_end = min(row_start + TILE_H, h)
                col_end = min(col_start + TILE_W, w)

                tile = image[row_start:row_end, col_start:col_end]
                actual_h, actual_w = tile.shape[:2]

                if actual_h != TILE_H or actual_w != TILE_W:
                    padded = np.zeros((TILE_H, TILE_W, 3), dtype=image.dtype)
                    padded[:actual_h, :actual_w] = tile
                    tile = padded

                tile_tensor = torch.from_numpy(tile).float().permute(2, 0, 1).unsqueeze(0) / 255.0
                tile_tensor = tile_tensor.to(DEVICE)

                output = model(tile_tensor)
                pred = output.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

                full_mask[row_start:row_end, col_start:col_end] = pred[:actual_h, :actual_w]

    return full_mask


def save_overlay(image: np.ndarray, mask: np.ndarray, class_legend: dict, out_path: Path):
    id_to_name = {v: k for k, v in class_legend.items()}
    color_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for class_id, name in id_to_name.items():
        color = CLASS_COLORS.get(name, (255, 255, 255))
        color_mask[mask == class_id] = color

    overlay = (0.6 * image + 0.4 * color_mask).astype(np.uint8)
    Image.fromarray(overlay).save(out_path)
    print(f"Saved overlay -> {out_path}")


def smooth_mask(mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    
    from scipy import ndimage

    smoothed = ndimage.median_filter(mask, size=kernel_size)
    return smoothed.astype(np.uint8)


def mask_to_geojson(mask: np.ndarray, transform, crs, class_legend: dict, out_path: Path,
                     min_polygon_area_px: int = 20):
    
    id_to_name = {v: k for k, v in class_legend.items()}
    background_id = class_legend.get("background", 0)

    
    pixel_area = abs(transform.a * transform.e)  
    min_area_crs_units = min_polygon_area_px * pixel_area

    records = []
    dropped_small = 0
    for geom_dict, class_id in rasterio_shapes(mask, transform=transform):
        class_id = int(class_id)
        if class_id == background_id:
            continue  
        geom = shapely_shape(geom_dict)
        if geom.area == 0:
            continue
        if geom.area < min_area_crs_units:
            dropped_small += 1
            continue
        records.append({
            "class_id": class_id,
            "class_name": id_to_name.get(class_id, f"unknown_{class_id}"),
            "geometry": geom,
        })

    if dropped_small:
        print(f"  Dropped {dropped_small} tiny noise polygon(s) "
              f"(< {min_polygon_area_px} pixels each)")

    if not records:
        print("  [!] No non-background regions found -- GeoJSON will be empty.")

    gdf = gpd.GeoDataFrame(records, crs=crs)
    gdf.to_file(out_path, driver="GeoJSON")
    print(f"Saved GeoJSON -> {out_path} ({len(gdf)} features)")
    return gdf


def main():
    parser = argparse.ArgumentParser(description="Run trained segmentation model on a new aerial image.")
    parser.add_argument("--image", required=True, type=str, help="Path to input GeoTIFF image")
    parser.add_argument("--weights", type=str, default=str(DEFAULT_WEIGHTS_PATH),
                         help="Path to trained model checkpoint")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR),
                         help="Directory to write outputs (mask, overlay, geojson)")
    args = parser.parse_args()

    image_path = Path(args.image)
    weights_path = Path(args.weights)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights not found: {weights_path}")

    print(f"Using device: {DEVICE}")
    model, class_legend = load_model(weights_path)
    num_classes = len(class_legend)

    print(f"\nReading image: {image_path}")
    src = rasterio.open(image_path)
    image = src.read([1, 2, 3]).transpose(1, 2, 0)
    transform = src.transform
    crs = src.crs
    src.close()
    print(f"  Image shape: {image.shape}, CRS: {crs}")

    if crs is None:
        print("  [!] WARNING: image has no CRS. GeoJSON output will lack real-world "
              "coordinates. Consider georeferencing this image first.")

    print("\nRunning inference (tiling image into 256x512 patches)...")
    raw_mask = run_inference_on_image(model, image, num_classes)

    print("Smoothing prediction (removing pixel-level noise)...")
    mask = smooth_mask(raw_mask, kernel_size=5)

    stem = image_path.stem
    mask_path = output_dir / f"{stem}_mask.png"
    Image.fromarray(mask).save(mask_path)
    print(f"Saved raw mask -> {mask_path}")

    overlay_path = output_dir / f"{stem}_overlay.png"
    save_overlay(image, mask, class_legend, overlay_path)

    geojson_path = output_dir / f"{stem}.geojson"
    mask_to_geojson(mask, transform, crs, class_legend, geojson_path)

    print(f"\nDone. All outputs saved to {output_dir}/")


if __name__ == "__main__":
    main()