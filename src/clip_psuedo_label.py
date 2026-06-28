import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import from_bounds as window_from_bounds
import geopandas as gpd
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GEOREF_DIR = PROJECT_ROOT / "data" / "processed" / "georeferenced"
LABELS_DIR = PROJECT_ROOT / "data" / "raw" / "labels" / "GeoJSON"
OUT_DIR = PROJECT_ROOT / "data" / "processed" / "pseudo_labels"
OUT_DIR.mkdir(parents=True, exist_ok=True)


DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {DEVICE}")


CLASS_PROMPTS = {
    "building": "an aerial photo of a building rooftop",
    "turf_grass": "an aerial photo of turf or mowed grass",
    "woody_vegetation": "an aerial photo of trees, tree canopy, or shrubs from above",
    "parking_lot": "an aerial photo of a paved parking lot with cars",
    "road": "an aerial photo of a paved road or street",
    "sidewalk": "an aerial photo of a narrow concrete sidewalk or path",
    "water": "an aerial photo of a body of water like a pool or pond",
}
CLASS_NAMES = list(CLASS_PROMPTS.keys())
PROMPTS = list(CLASS_PROMPTS.values())

print("Loading CLIP model (first run will download weights, ~600MB)...")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model.eval()

def _extract_features(output):
    """Some transformers versions return a raw tensor from
    get_text_features/get_image_features, others return a wrapper
    object with the tensor inside. Handle both."""
    if torch.is_tensor(output):
        return output
    
    for attr in ("text_embeds", "image_embeds", "pooler_output", "last_hidden_state"):
        if hasattr(output, attr):
            return getattr(output, attr)
    raise TypeError(f"Unexpected output type from CLIP feature call: {type(output)}")



with torch.no_grad():
    text_inputs = processor(text=PROMPTS, return_tensors="pt", padding=True).to(DEVICE)
    text_features = _extract_features(model.get_text_features(**text_inputs))
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)


def classify_crop(crop_img: Image.Image):
    """Returns (predicted_class_name, confidence_float, all_scores_dict)."""
    with torch.no_grad():
        image_inputs = processor(images=crop_img, return_tensors="pt").to(DEVICE)
        image_features = _extract_features(model.get_image_features(**image_inputs))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        
        logits = (image_features @ text_features.T) * model.logit_scale.exp()
        probs = logits.softmax(dim=-1).cpu().numpy()[0]

    best_idx = int(np.argmax(probs))
    scores = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}
    return CLASS_NAMES[best_idx], float(probs[best_idx]), scores


def process_parcel(parcel_id: int):
    image_path = GEOREF_DIR / f"georef_{parcel_id}.tiff"
    label_path = LABELS_DIR / f"{parcel_id}.geojson"

    print(f"\n--- Parcel {parcel_id} ---")
    gdf = gpd.read_file(label_path)
    print(f"Loaded {len(gdf)} polygons")

    src = rasterio.open(image_path)
    if gdf.crs != src.crs:
        gdf = gdf.to_crs(src.crs)

    results = []
    for idx, row in gdf.iterrows():
        geom = row.geometry
        minx, miny, maxx, maxy = geom.bounds

        
        w = maxx - minx
        h = maxy - miny
        pad_x = w * 0.1 if w > 0 else 1e-6
        pad_y = h * 0.1 if h > 0 else 1e-6

        window = window_from_bounds(
            minx - pad_x, miny - pad_y, maxx + pad_x, maxy + pad_y,
            transform=src.transform
        )

        try:
            patch = src.read([1, 2, 3], window=window)
        except Exception as e:
            print(f"  [!] feature {idx}: failed to read window ({e}), skipping")
            continue

        if patch.size == 0 or patch.shape[1] == 0 or patch.shape[2] == 0:
            print(f"  [!] feature {idx}: empty crop, skipping")
            continue

        patch_img = Image.fromarray(patch.transpose(1, 2, 0))

        pred_class, confidence, all_scores = classify_crop(patch_img)

        results.append({
            "feature_idx": idx,
            "pred_class": pred_class,
            "confidence": confidence,
            "area": geom.area,
        })

        if idx % 10 == 0:
            print(f"  feature {idx}/{len(gdf)}: {pred_class} ({confidence:.2f})")

    gdf["pred_class"] = [r["pred_class"] for r in results]
    gdf["confidence"] = [r["confidence"] for r in results]

    out_geojson = OUT_DIR / f"{parcel_id}_pseudo_labeled.geojson"
    gdf.to_file(out_geojson, driver="GeoJSON")
    print(f"Saved labeled GeoJSON -> {out_geojson}")

    
    csv_path = OUT_DIR / f"{parcel_id}_pseudo_labels_for_review.csv"
    gdf_sorted = gdf[["pred_class", "confidence"]].copy()
    gdf_sorted["feature_idx"] = gdf_sorted.index
    gdf_sorted = gdf_sorted.sort_values("confidence")
    gdf_sorted.to_csv(csv_path, index=False)
    print(f"Saved review CSV -> {csv_path}")

    src.close()
    return gdf


if __name__ == "__main__":
    for parcel_id in [1, 2, 3]:
        process_parcel(parcel_id)

    print("\nDone. Review the *_pseudo_labels_for_review.csv files,")
    print("starting with the lowest-confidence rows, before trusting these labels.")