import os
from pathlib import Path

import pandas as pd

D = Path(__file__).resolve().parents[1] / "data" / "milk10k"
files = [os.path.join(r, f) for r, _, fs in os.walk(D / "images") for f in fs]
print("Image files:", len(files))
print("Examples:", files[:3])

for name in ["Metadata", "Supplement", "GroundTruth"]:
    df = pd.read_csv(D / f"MILK10k_Training_{name}.csv")
    print(f"\n{name}: {df.shape}")
    print(list(df.columns))
    print(df.head(2).T)