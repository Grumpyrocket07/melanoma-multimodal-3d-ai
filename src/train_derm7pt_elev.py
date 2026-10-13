import argparse
import json

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms as T

from derm7pt_data import PairedDermDataset, load_derm7pt_splits
from train_derm7pt import MEAN, OUT_DIR, SIZE, STD, make_encoder, sens_spec

ELEV = {"flat": 0, "palpable": 1, "nodular": 2}


class PairedElevDataset(PairedDermDataset):
    """One item = (clinical image, dermoscopic image, elevation one-hot, label)."""

    def __getitem__(self, i):
        clinic, derm, label = super().__getitem__(i)
        elev = torch.zeros(3)
        elev[ELEV[self.df.iloc[i]["elevation"]]] = 1.0
        return clinic, derm, elev, label


class FusionNet(nn.Module):
    def __init__(self, use_elev):
        super().__init__()
        self.use_elev = use_elev
        self.clinic = make_encoder("imagenet")
        self.derm = make_encoder("imagenet")
        dim = self.clinic.num_features + self.derm.num_features
        self.img_proj = nn.Sequential(nn.Dropout(0.3), nn.Linear(dim, 64), nn.ReLU())
        out_dim = 64
        if use_elev:
            self.elev_proj = nn.Sequential(nn.Linear(3, 16), nn.ReLU())
            out_dim += 16
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(out_dim, 1))

    def forward(self, clinic, derm, elev):
        x = self.img_proj(torch.cat([self.clinic(clinic), self.derm(derm)], dim=1))
        if self.use_elev:
            x = torch.cat([x, self.elev_proj(elev)], dim=1)
        return self.head(x).squeeze(1)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs, labels = [], []
    for clinic, derm, elev, y in loader:
        clinic = clinic.to(device, non_blocking=True)
        derm = derm.to(device, non_blocking=True)
        elev = elev.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(clinic, derm, elev)
        probs.append(torch.sigmoid(logits.float()).cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(probs), np.concatenate(labels)


def test_metrics(val_probs, val_labels, test_probs, test_labels):
    fpr, tpr, thr = roc_curve(val_labels, val_probs)
    threshold = float(thr[np.argmax(tpr - fpr)])
    sens, spec = sens_spec(test_probs, test_labels, threshold)
    return threshold, float(roc_auc_score(test_labels, test_probs)), sens, spec


def run_seed(args, seed, device, dfs, train_tf, eval_tf):
    train_df, val_df, test_df = dfs
    torch.manual_seed(seed)
    np.random.seed(seed)

    kw = dict(num_workers=4, pin_memory=True, persistent_workers=True)
    train_loader = DataLoader(PairedElevDataset(train_df, train_tf, train_tf),
                              batch_size=args.batch_size, shuffle=True, drop_last=True, **kw)
    val_loader = DataLoader(PairedElevDataset(val_df, eval_tf, eval_tf), batch_size=32, shuffle=False, **kw)
    test_loader = DataLoader(PairedElevDataset(test_df, eval_tf, eval_tf), batch_size=32, shuffle=False, **kw)

    model = FusionNet(use_elev=(args.mode == "both_elev")).to(device)
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
        for clinic, derm, elev, y in train_loader:
            clinic = clinic.to(device, non_blocking=True)
            derm = derm.to(device, non_blocking=True)
            elev = elev.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(clinic, derm, elev)
            loss = criterion(logits.float(), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += loss.item()
        scheduler.step()

        val_probs, val_labels = predict(model, val_loader, device)
        val_auc = float(roc_auc_score(val_labels, val_probs))
        print(f"  seed {seed} epoch {epoch}/{args.epochs}  loss {total / len(train_loader):.4f}  val_AUC {val_auc:.4f}")
        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    val_probs, val_labels = predict(model, val_loader, device)
    test_probs, test_labels = predict(model, test_loader, device)
    threshold, test_auc, sens, spec = test_metrics(val_probs, val_labels, test_probs, test_labels)
    print(f"seed {seed}: TEST AUC {test_auc:.4f}  sens {sens:.3f}  spec {spec:.3f}")
    return {"seed": seed, "val_auc_best": best_auc, "threshold_from_val": threshold,
            "test_auc": test_auc, "test_sensitivity": sens, "test_specificity": spec}


def run_elev_only(dfs):
    """Elevation label alone, no images: logistic regression on the 3 categories."""
    train_df, val_df, test_df = dfs

    def feats(df):
        return np.eye(3)[df["elevation"].map(ELEV).to_numpy()]

    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(feats(train_df), train_df["is_melanoma"])
    val_probs = clf.predict_proba(feats(val_df))[:, 1]
    test_probs = clf.predict_proba(feats(test_df))[:, 1]
    val_labels = val_df["is_melanoma"].to_numpy()
    test_labels = test_df["is_melanoma"].to_numpy()
    threshold, test_auc, sens, spec = test_metrics(val_probs, val_labels, test_probs, test_labels)
    return {"seed": None, "val_auc_best": float(roc_auc_score(val_labels, val_probs)),
            "threshold_from_val": threshold, "test_auc": test_auc,
            "test_sensitivity": sens, "test_specificity": spec}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["elev", "both", "both_elev"], required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dfs = load_derm7pt_splits()
    for d in dfs:
        assert d["elevation"].isin(ELEV).all(), "Unexpected elevation value"

    if args.mode == "elev":
        runs = [run_elev_only(dfs)]
        tag = "elev_only"
    else:
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
        train_tf = T.Compose([
            T.Resize((SIZE, SIZE)), T.RandomHorizontalFlip(), T.RandomVerticalFlip(), T.RandomRotation(20),
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1), T.ToTensor(), T.Normalize(MEAN, STD),
        ])
        eval_tf = T.Compose([T.Resize((SIZE, SIZE)), T.ToTensor(), T.Normalize(MEAN, STD)])
        print(f"Mode: {args.mode}  seeds: {args.seeds}")
        runs = [run_seed(args, s, device, dfs, train_tf, eval_tf) for s in args.seeds]
        tag = f"fusion_{args.mode}"

    summary = {}
    for key in ["val_auc_best", "test_auc", "test_sensitivity", "test_specificity"]:
        vals = np.array([r[key] for r in runs])
        summary[key] = {"mean": float(vals.mean()), "std": float(vals.std())}
    print(f"\nSUMMARY ({args.mode}, {len(runs)} run(s))")
    for key, v in summary.items():
        print(f"  {key}: {v['mean']:.3f} +/- {v['std']:.3f}")

    with open(OUT_DIR / f"{tag}.json", "w") as f:
        json.dump({"args": vars(args), "summary": summary, "runs": runs}, f, indent=2)


if __name__ == "__main__":
    main()