import json
from pathlib import Path

import albumentations as A
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TILES_DIR = PROJECT_ROOT / "data" / "processed" / "tiles"

with open(TILES_DIR / "class_legend.json") as f:
    CLASS_LEGEND = json.load(f)
NUM_CLASSES = len(CLASS_LEGEND)

VAL_PARCEL_PREFIX = "1_"  


def get_train_val_files():
    all_images = sorted((TILES_DIR / "images").glob("*.png"))
    train_files = [p for p in all_images if not p.name.startswith(VAL_PARCEL_PREFIX)]
    val_files = [p for p in all_images if p.name.startswith(VAL_PARCEL_PREFIX)]
    return train_files, val_files


def get_train_augmentations():
    
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.HueSaturationValue(p=0.2),
        A.GaussNoise(p=0.1),
    ])


def get_val_augmentations():
    return None  


class TileSegmentationDataset(Dataset):
    def __init__(self, file_list, augmentations=None):
        self.file_list = file_list
        self.augmentations = augmentations

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        mask_path = TILES_DIR / "masks" / img_path.name

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path))

        if self.augmentations is not None:
            augmented = self.augmentations(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]

        
        image = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
        mask = torch.from_numpy(mask).long()

        return image, mask


def compute_tile_sample_weights(file_list, class_weights):
    weights = []
    for img_path in file_list:
        mask_path = TILES_DIR / "masks" / img_path.name
        mask = np.array(Image.open(mask_path))
        present_classes = np.unique(mask)
        tile_weight = max(class_weights[c].item() for c in present_classes)
        weights.append(tile_weight)
    return weights


def compute_class_weights():
    train_files, _ = get_train_val_files()
    counts = np.zeros(NUM_CLASSES, dtype=np.int64)

    for img_path in train_files:
        mask_path = TILES_DIR / "masks" / img_path.name
        mask = np.array(Image.open(mask_path))
        for c in range(NUM_CLASSES):
            counts[c] += np.sum(mask == c)

    
    counts = np.maximum(counts, 1)
    weights = 1.0 / counts
    weights = weights / weights.sum() * NUM_CLASSES  
    return torch.tensor(weights, dtype=torch.float32), counts


if __name__ == "__main__":
    train_files, val_files = get_train_val_files()
    print(f"Train tiles: {len(train_files)} (parcels 2+3)")
    print(f"Val tiles: {len(val_files)} (parcel 1)")

    weights, counts = compute_class_weights()
    id_to_name = {v: k for k, v in CLASS_LEGEND.items()}
    print("\nPer-class pixel counts (training set) and computed loss weights:")
    for c in range(NUM_CLASSES):
        print(f"  {id_to_name[c]:20s} count={counts[c]:>12,}  weight={weights[c]:.3f}")