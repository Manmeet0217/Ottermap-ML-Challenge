import json
import sys
from pathlib import Path

import numpy as np
import segmentation_models_pytorch as smp
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dataset import get_train_val_files, CLASS_LEGEND  

TILES_DIR = PROJECT_ROOT / "data" / "processed" / "tiles"
WEIGHTS_PATH = PROJECT_ROOT / "weights" / "best_model.pth"
RESULTS_DIR = PROJECT_ROOT / "results"

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

N_SAMPLES_PER_SET = 5


def load_model():
    checkpoint = torch.load(WEIGHTS_PATH, map_location=DEVICE, weights_only=False)
    model = smp.Unet(
        encoder_name=checkpoint.get("encoder_name", "resnet34"),
        encoder_weights=None,
        in_channels=3,
        classes=checkpoint["num_classes"],
    ).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint["class_legend"]


def predict_tile(model, image: np.ndarray):
    tensor = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    tensor = tensor.to(DEVICE)
    with torch.no_grad():
        output = model(tensor)
    pred = output.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    return pred


def make_overlay(image: np.ndarray, mask: np.ndarray, class_legend: dict):
    id_to_name = {v: k for k, v in class_legend.items()}
    color_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for class_id, name in id_to_name.items():
        color_mask[mask == class_id] = CLASS_COLORS.get(name, (255, 255, 255))
    return (0.6 * image + 0.4 * color_mask).astype(np.uint8)


def make_side_by_side(image: np.ndarray, gt_mask: np.ndarray, pred_overlay: np.ndarray,
                       class_legend: dict):
    gt_overlay = make_overlay(image, gt_mask, class_legend)
    combined = np.concatenate([image, gt_overlay, pred_overlay], axis=1)
    return combined


def process_set(model, class_legend, file_list, out_dir: Path, set_name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    n = min(N_SAMPLES_PER_SET, len(file_list))
    # Spread samples across the list rather than just taking the first N
    indices = np.linspace(0, len(file_list) - 1, n, dtype=int)

    print(f"\n--- {set_name}: generating {n} sample overlays ---")
    for i in indices:
        img_path = file_list[i]
        mask_path = TILES_DIR / "masks" / img_path.name

        image = np.array(Image.open(img_path).convert("RGB"))
        gt_mask = np.array(Image.open(mask_path))

        pred_mask = predict_tile(model, image)
        pred_overlay = make_overlay(image, pred_mask, class_legend)
        combined = make_side_by_side(image, gt_mask, pred_overlay, class_legend)

        out_path = out_dir / f"{img_path.stem}_comparison.png"
        Image.fromarray(combined).save(out_path)
        print(f"  Saved {out_path.name} (left=image, center=ground truth, right=prediction)")


if __name__ == "__main__":
    model, class_legend = load_model()
    train_files, val_files = get_train_val_files()

    process_set(model, class_legend, train_files,
                RESULTS_DIR / "training_predictions", "Training set")
    process_set(model, class_legend, val_files,
                RESULTS_DIR / "validation_predictions", "Validation set")

    print(f"\nDone. Results saved under {RESULTS_DIR}/")
    print("External predictions: run inference.py directly on each file in "
          "data/external_test/, then copy the *_overlay.png outputs into "
          "results/external_predictions/")