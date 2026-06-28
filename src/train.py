import json
import time
from pathlib import Path

import numpy as np
import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from dataset import (
    CLASS_LEGEND, NUM_CLASSES, TileSegmentationDataset,
    compute_class_weights, compute_tile_sample_weights,
    get_train_augmentations, get_train_val_files, get_val_augmentations,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = PROJECT_ROOT / "weights"
REPORTS_DIR = PROJECT_ROOT / "reports"
WEIGHTS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {DEVICE}")

BATCH_SIZE = 4          
NUM_EPOCHS = 40
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4      
FREEZE_ENCODER_EPOCHS = 10  
EARLY_STOPPING_PATIENCE = 10  
ID_TO_NAME = {v: k for k, v in CLASS_LEGEND.items()}


def compute_iou_per_class(pred, target, num_classes):
    
    ious = np.full(num_classes, np.nan)
    pred = pred.flatten()
    target = target.flatten()
    for c in range(num_classes):
        pred_c = pred == c
        target_c = target == c
        intersection = (pred_c & target_c).sum().item()
        union = (pred_c | target_c).sum().item()
        if union > 0:
            ious[c] = intersection / union
    return ious


def main():
    train_files, val_files = get_train_val_files()
    print(f"Train tiles: {len(train_files)} | Val tiles: {len(val_files)}")
    if len(train_files) == 0 or len(val_files) == 0:
        raise RuntimeError(
            "Train or val set is empty -- check that tiles exist in "
            "data/processed/tiles/images/ and that parcel 1 tiles "
            "(prefix '1_') are present for validation."
        )

    class_weights, class_counts = compute_class_weights()
    class_weights = class_weights.to(DEVICE)
    print("\nClass weights (inverse frequency, normalized):")
    for c in range(NUM_CLASSES):
        print(f"  {ID_TO_NAME[c]:20s} weight={class_weights[c].item():.3f}  train_pixel_count={class_counts[c]:,}")

    train_ds = TileSegmentationDataset(train_files, augmentations=get_train_augmentations())
    val_ds = TileSegmentationDataset(val_files, augmentations=get_val_augmentations())

    
    tile_weights = compute_tile_sample_weights(train_files, class_weights.cpu())
    sampler = WeightedRandomSampler(tile_weights, num_samples=len(train_files), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",  
        in_channels=3,
        classes=NUM_CLASSES,
    ).to(DEVICE)

    def set_encoder_trainable(trainable: bool):
        for param in model.encoder.parameters():
            param.requires_grad = trainable

    set_encoder_trainable(False)
    print(f"\nEncoder frozen for the first {FREEZE_ENCODER_EPOCHS} epochs "
          f"(training decoder only) to reduce overfitting on this small dataset.")

    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    best_miou = -1.0
    epochs_since_improvement = 0
    history = []

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()

        if epoch == FREEZE_ENCODER_EPOCHS + 1:
            set_encoder_trainable(True)
            print(f"  [epoch {epoch}] Unfreezing encoder for full fine-tuning")

        
        model.train()
        train_loss_total = 0.0
        for images, masks in train_loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss_total += loss.item() * images.size(0)
        train_loss = train_loss_total / len(train_ds)

        
        model.eval()
        val_loss_total = 0.0
        all_ious = []
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(DEVICE), masks.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss_total += loss.item() * images.size(0)

                preds = outputs.argmax(dim=1)
                batch_ious = compute_iou_per_class(preds.cpu(), masks.cpu(), NUM_CLASSES)
                all_ious.append(batch_ious)

        val_loss = val_loss_total / len(val_ds)

        all_ious = np.array(all_ious)  
        per_class_iou = np.nanmean(all_ious, axis=0)  
        miou = np.nanmean(per_class_iou)

        scheduler.step(val_loss)
        elapsed = time.time() - t0

        print(f"Epoch {epoch:3d}/{NUM_EPOCHS} | train_loss={train_loss:.4f} "
              f"val_loss={val_loss:.4f} mIoU={miou:.4f} ({elapsed:.1f}s)")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "mIoU": float(miou),
            "per_class_iou": {ID_TO_NAME[c]: (None if np.isnan(per_class_iou[c]) else float(per_class_iou[c]))
                               for c in range(NUM_CLASSES)},
        })

        if miou > best_miou:
            best_miou = miou
            epochs_since_improvement = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "encoder_name": "resnet34",
                "num_classes": NUM_CLASSES,
                "class_legend": CLASS_LEGEND,
                "epoch": epoch,
                "val_mIoU": miou,
            }, WEIGHTS_DIR / "best_model.pth")
            print(f"  -> New best mIoU ({miou:.4f}), saved checkpoint")
        else:
            epochs_since_improvement += 1
            if epochs_since_improvement >= EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping: no val mIoU improvement in "
                      f"{EARLY_STOPPING_PATIENCE} epochs (best={best_miou:.4f} at "
                      f"epoch {epoch - epochs_since_improvement}).")
                break

    with open(REPORTS_DIR / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete. Best val mIoU: {best_miou:.4f}")
    print(f"Best checkpoint: {WEIGHTS_DIR / 'best_model.pth'}")
    print(f"Full training history: {REPORTS_DIR / 'training_history.json'}")


if __name__ == "__main__":
    main()