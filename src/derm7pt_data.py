from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data" / "derm7pt" / "release_v0"
IMG_DIR = BASE / "images"


def load_derm7pt_splits():
    """Returns train, val, test DataFrames using the dataset's suggested split."""
    meta = pd.read_csv(BASE / "meta" / "meta.csv")
    meta["is_melanoma"] = meta["diagnosis"].str.startswith("melanoma").astype(int)

    names = ["train", "valid", "test"]
    idx = {n: pd.read_csv(BASE / "meta" / f"{n}_indexes.csv")["indexes"].to_numpy() for n in names}
    all_idx = np.concatenate([idx[n] for n in names])
    n_rows = len(meta)
    if len(set(all_idx)) == len(all_idx) == n_rows and set(all_idx) == set(range(n_rows)):
        offset = 0
    elif len(set(all_idx)) == len(all_idx) == n_rows and set(all_idx) == set(range(1, n_rows + 1)):
        offset = 1
    else:
        raise ValueError("Index files do not cleanly split the metadata rows.")
    train, val, test = [meta.iloc[idx[n] - offset].reset_index(drop=True) for n in names]
    return train, val, test


class PairedDermDataset(Dataset):
    """One item = one case: (clinical image, dermoscopic image, label)."""

    def __init__(self, df, clinic_tf, derm_tf):
        self.df = df.reset_index(drop=True)
        self.clinic_tf = clinic_tf
        self.derm_tf = derm_tf

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        clinic = Image.open(IMG_DIR / row["clinic"]).convert("RGB")
        derm = Image.open(IMG_DIR / row["derm"]).convert("RGB")
        label = torch.tensor(float(row["is_melanoma"]), dtype=torch.float32)
        return self.clinic_tf(clinic), self.derm_tf(derm), label


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    train, val, test = load_derm7pt_splits()
    for name, d in [("train", train), ("val", val), ("test", test)]:
        print(f"{name}: {len(d)} cases, {int(d['is_melanoma'].sum())} melanoma "
              f"({d['is_melanoma'].mean() * 100:.1f}%)")

    sample = train.sample(3, random_state=0)
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for j, (_, row) in enumerate(sample.iterrows()):
        axes[0, j].imshow(Image.open(IMG_DIR / row["clinic"]).convert("RGB"))
        axes[0, j].set_title(f"clinical: {row['diagnosis']}", fontsize=9)
        axes[1, j].imshow(Image.open(IMG_DIR / row["derm"]).convert("RGB"))
        axes[1, j].set_title("dermoscopic", fontsize=9)
    for ax in axes.flat:
        ax.axis("off")
    out = BASE.parent / "sample_pairs.png"
    plt.tight_layout()
    plt.savefig(out, dpi=100)
    print("Saved sample figure to", out)