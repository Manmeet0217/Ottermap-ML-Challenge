from pathlib import Path

import matplotlib.pyplot as plt
import rasterio

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXTERNAL_DIR = PROJECT_ROOT / "data" / "external_test"

names = ["austin_tx_residential", "denver_co_residential", "miami_fl_residential"]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for ax, name in zip(axes, names):
    src = rasterio.open(EXTERNAL_DIR / f"{name}.tif")
    image = src.read([1, 2, 3]).transpose(1, 2, 0)
    ax.imshow(image)
    ax.set_title(name)
    ax.axis("off")
    src.close()

plt.tight_layout()
plt.show()