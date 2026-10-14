import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torchvision import transforms as T

from milk10k_data import load_milk10k_splits
from train_derm7pt import MEAN, SIZE, STD, run_seed

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "milk10k"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["derm", "clinic", "both"], required=True)
    parser.add_argument("--derm_init", choices=["imagenet", "ham"], default="imagenet")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()

    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train_tf = T.Compose([
        T.Resize((SIZE, SIZE)), T.RandomHorizontalFlip(), T.RandomVerticalFlip(), T.RandomRotation(20),
        T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1), T.ToTensor(), T.Normalize(MEAN, STD),
    ])
    eval_tf = T.Compose([T.Resize((SIZE, SIZE)), T.ToTensor(), T.Normalize(MEAN, STD)])
    dfs = load_milk10k_splits()

    print(f"MILK10k  mode: {args.mode}  derm_init: {args.derm_init}  seeds: {args.seeds}")
    runs = [run_seed(args, s, device, dfs, train_tf, eval_tf) for s in args.seeds]

    summary = {}
    for key in ["val_auc_best", "test_auc", "test_sensitivity", "test_specificity"]:
        vals = np.array([r[key] for r in runs])
        summary[key] = {"mean": float(vals.mean()), "std": float(vals.std())}
    print(f"\nSUMMARY (MILK10k, {args.mode}, derm_init={args.derm_init}, {len(runs)} seeds)")
    for key, v in summary.items():
        print(f"  {key}: {v['mean']:.3f} +/- {v['std']:.3f}")

    tag = args.mode if args.mode == "clinic" else f"{args.mode}_{args.derm_init}"
    with open(OUT_DIR / f"{tag}.json", "w") as f:
        json.dump({"args": vars(args), "summary": summary, "runs": runs}, f, indent=2)


if __name__ == "__main__":
    main()