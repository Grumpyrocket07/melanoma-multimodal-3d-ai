from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class HamDataset(Dataset):
    """Loads HAM10000 images listed in a split CSV. Label: 1 = melanoma, 0 = other."""

    def __init__(self, csv_path, img_dir, transform=None):
        self.df = pd.read_csv(csv_path)
        self.img_dir = Path(img_dir)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(self.img_dir / f"{row['image_id']}.jpg").convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        label = torch.tensor(float(row["is_melanoma"]), dtype=torch.float32)
        return img, label