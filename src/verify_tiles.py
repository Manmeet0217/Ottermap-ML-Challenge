import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TILES_DIR = PROJECT_ROOT / "data" / "processed" / "tiles"

with open(TILES_DIR / "class_legend.json") as f:
    legend = json.load(f)
id_to_name = {v: k for k, v in legend.items()}

image_files = sorted((TILES_DIR / "images").glob("*.png"))
sample = random.sample(image_files, min(4, len(image_files)))

fig, axes = plt.subplots(2, len(sample), figsize=(5 * len(sample), 8))

for col, img_path in enumerate(sample):
    mask_path = TILES_DIR / "masks" / img_path.name

    img = np.array(Image.open(img_path))
    mask = np.array(Image.open(mask_path))

    axes[0, col].imshow(img)
    axes[0, col].set_title(img_path.name)
    axes[0, col].axis("off")

    axes[1, col].imshow(mask, cmap="tab10", vmin=0, vmax=9)
    present = sorted(set(np.unique(mask).tolist()))
    names = [id_to_name.get(c, str(c)) for c in present]
    axes[1, col].set_title("mask: " + ", ".join(names), fontsize=9)
    axes[1, col].axis("off")

plt.tight_layout()
plt.show()