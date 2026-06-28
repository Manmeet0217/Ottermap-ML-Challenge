from pathlib import Path
import geopandas as gpd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LABEL_DIR = PROJECT_ROOT / "data/raw/labels/GeoJSON"

for file in sorted(LABEL_DIR.glob("*.geojson")):
    print("\n" + "=" * 80)
    print(file.name)

    gdf = gpd.read_file(file)

    print(gdf.head())
    print("\nColumns:")
    print(gdf.columns)
    print("\nGeometry Types:")
    print(gdf.geom_type.value_counts())