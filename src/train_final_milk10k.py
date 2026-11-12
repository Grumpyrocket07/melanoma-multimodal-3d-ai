import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms as T

from derm7pt_data import PairedDermDataset
from milk10k_data import load_milk10k_splits
from train_derm7pt import MEAN, SIZE, STD, Net, predict, sens_spec

ROOT = Path(__file__).resolve().parents[1]


def train_one(args, mode, seed, device, dfs, train_tf, eval_tf, out_dir):
    train_df, val_df, test_df = dfs
    torch.manual_seed(seed)
    np.random.seed(seed)

    kw = dict(num_workers=4, pin_memory=True, persistent_workers=True)
    train_loader = DataLoader(PairedDermDataset(train_df, train_tf, train_tf),
                              batch_size=args.batch_size, shuffle=True, drop_last=True, **kw)
    val_loader = DataLoader(PairedDermDataset(val_df, eval_tf, eval_tf), batch_size=32, shuffle=False, **kw)
    test_loader = DataLoader(PairedDermDataset(test_df, eval_tf, eval_tf), batch_size=32, shuffle=False, **kw)

    model = Net(mode, "imagenet").to(device)
    n_pos = train_df["is_melanoma"].sum()
    pos_weight = torch.tensor((len(train_df) - n_pos) / n_pos, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda")

    best_auc, best_state = -1.0, None
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for clinic, derm, y in train_loader:
            clinic = clinic.to(device, non_blocking=True)
            derm = derm.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(clinic, derm)
            loss = criterion(logits.float(), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += loss.item()
        scheduler.step()

        val_probs, val_labels = predict(model, val_loader, device)
        val_auc = float(roc_auc_score(val_labels, val_probs))
        print(f"  {mode} seed {seed} epoch {epoch}/{args.epochs}  loss {total / len(train_loader):.4f}  val_AUC {val_auc:.4f}")
        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    torch.save(best_state, out_dir / f"{mode}_seed{seed}.pt")
    model.load_state_dict(best_state)
    val_probs, val_labels = predict(model, val_loader, device)
    test_probs, test_labels = predict(model, test_loader, device)

    fpr, tpr, thr = roc_curve(val_labels, val_probs)
    youden = float(thr[np.argmax(tpr - fpr)])
    low_thr = float(np.quantile(val_probs[val_labels == 1], 0.05))
    high_thr = float(np.quantile(val_probs[val_labels == 0], 0.90))
    sens, spec = sens_spec(test_probs, test_labels, youden)
    info = {"mode": mode, "seed": seed, "epochs": args.epochs, "val_auc_best": best_auc,
            "test_auc": float(roc_auc_score(test_labels, test_probs)),
            "threshold_youden": youden, "test_sensitivity": sens, "test_specificity": spec,
            "risk_low_below": low_thr, "risk_high_at_or_above": high_thr}
    with open(out_dir / f"{mode}_seed{seed}.json", "w") as f:
        json.dump(info, f, indent=2)
    print(f"SAVED {mode} seed {seed}: test AUC {info['test_auc']:.4f}  "
          f"risk cutoffs low < {low_thr:.3f}, high >= {high_thr:.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", nargs="+", choices=["derm", "clinic", "both"], default=["clinic", "derm", "both"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--out_dir", default="milk10k_final")
    args = parser.parse_args()

    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    out_dir = ROOT / "results" / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    train_tf = T.Compose([
        T.Resize((SIZE, SIZE)), T.RandomHorizontalFlip(), T.RandomVerticalFlip(), T.RandomRotation(20),
        T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1), T.ToTensor(), T.Normalize(MEAN, STD),
    ])
    eval_tf = T.Compose([T.Resize((SIZE, SIZE)), T.ToTensor(), T.Normalize(MEAN, STD)])
    dfs = load_milk10k_splits()

    for mode in args.modes:
        for seed in args.seeds:
            train_one(args, mode, seed, device, dfs, train_tf, eval_tf, out_dir)


if __name__ == "__main__":
    main()