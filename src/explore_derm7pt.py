from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data" / "derm7pt" / "release_v0"
meta = pd.read_csv(BASE / "meta" / "meta.csv")

print("Cases:", len(meta))
print("\nDiagnosis counts:")
print(meta["diagnosis"].value_counts())

for name in ["train", "valid", "test"]:
    idx = pd.read_csv(BASE / "meta" / f"{name}_indexes.csv")
    print(f"\n{name}_indexes.csv:", idx.shape, list(idx.columns))
    print(idx.head(3))

for col in ["clinic", "derm"]:
    missing = [p for p in meta[col] if not (BASE / "images" / p).exists()]
    print(f"\nMissing {col} images:", len(missing))
    print(f"Duplicate paths in {col}:", int(meta[col].duplicated().sum()))
print("\nCases using the same file for clinic and derm:", int((meta["clinic"] == meta["derm"]).sum()))
print("Sex:", meta["sex"].value_counts().to_dict())
print("Elevation:", meta["elevation"].value_counts().to_dict())