import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from torchvision import transforms as T

from milk10k_data import load_milk10k_splits
from train_derm7pt import MEAN, SIZE, STD, run_seed

ROOT = Path(__file__).resolve().parents[1]
SEED = 42


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", nargs="+", default=["derm", "clinic", "both"])
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dir", default="milk10k_cv")
    args = parser.parse_args()
    args.derm_init = "imagenet"

    out_dir = ROOT / "results" / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True

    train_tf = T.Compose([
        T.Resize((SIZE, SIZE)), T.RandomHorizontalFlip(), T.RandomVerticalFlip(), T.RandomRotation(20),
        T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1), T.ToTensor(), T.Normalize(MEAN, STD),
    ])
    eval_tf = T.Compose([T.Resize((SIZE, SIZE)), T.ToTensor(), T.Normalize(MEAN, STD)])

    full = pd.concat(load_milk10k_splits(), ignore_index=True)
    print(f"Total lesions: {len(full)}  melanoma: {int(full['is_melanoma'].sum())}")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    for fold, (tr_idx, te_idx) in enumerate(skf.split(full, full["dx"])):
        if fold not in args.folds:
            continue
        rest = full.iloc[tr_idx]
        test_df = full.iloc[te_idx].reset_index(drop=True)
        train_df, val_df = train_test_split(rest, test_size=0.15, stratify=rest["dx"], random_state=SEED + fold)
        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)
        print(f"\n=== Fold {fold}: train {len(train_df)} / val {len(val_df)} / test {len(test_df)} "
              f"(test melanoma {int(test_df['is_melanoma'].sum())}) ===")

        for mode in args.modes:
            out = out_dir / f"fold{fold}_{mode}.json"
            if out.exists():
                print(f"fold {fold} {mode}: already done, skipping")
                continue
            args.mode = mode
            print(f"--- fold {fold}, mode {mode} ---")
            run = run_seed(args, args.seed + fold, device, (train_df, val_df, test_df), train_tf, eval_tf)
            run["fold"] = fold
            run["mode"] = mode
            run["lesion_ids"] = test_df["lesion_id"].tolist()
            run["labels"] = test_df["is_melanoma"].astype(int).tolist()
            with open(out, "w") as f:
                json.dump(run, f)


if __name__ == "__main__":
    main()