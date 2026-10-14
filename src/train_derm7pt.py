import argparse
import json
from pathlib import Path

import numpy as np
import timm
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms as T

from derm7pt_data import PairedDermDataset, load_derm7pt_splits

ROOT = Path(__file__).resolve().parents[1]
HAM_CKPT = ROOT / "results" / "baseline_efficientnet_b0" / "best.pt"
OUT_DIR = ROOT / "results" / "derm7pt"
SIZE = 224
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def make_encoder(init):
    enc = timm.create_model("efficientnet_b0", pretrained=True, num_classes=0)
    if init == "ham":
        state = torch.load(HAM_CKPT, map_location="cpu")
        missing, unexpected = enc.load_state_dict(state, strict=False)
        assert not missing, f"Missing keys: {missing}"
    return enc


class Net(nn.Module):
    def __init__(self, mode, derm_init):
        super().__init__()
        self.encoders = nn.ModuleDict()
        if mode in ("derm", "both"):
            self.encoders["derm"] = make_encoder(derm_init)
        if mode in ("clinic", "both"):
            self.encoders["clinic"] = make_encoder("imagenet")
        dim = sum(e.num_features for e in self.encoders.values())
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(dim, 1))

    def forward(self, clinic, derm):
        feats = []
        if "clinic" in self.encoders:
            feats.append(self.encoders["clinic"](clinic))
        if "derm" in self.encoders:
            feats.append(self.encoders["derm"](derm))
        return self.head(torch.cat(feats, dim=1)).squeeze(1)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs, labels = [], []
    for clinic, derm, y in loader:
        clinic = clinic.to(device, non_blocking=True)
        derm = derm.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(clinic, derm)
        probs.append(torch.sigmoid(logits.float()).cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(probs), np.concatenate(labels)


def sens_spec(probs, labels, thr):
    pred = probs >= thr
    tp = int(((pred == 1) & (labels == 1)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    return tp / (tp + fn), tn / (tn + fp)


def run_seed(args, seed, device, dfs, train_tf, eval_tf):
    train_df, val_df, test_df = dfs
    torch.manual_seed(seed)
    np.random.seed(seed)

    kw = dict(num_workers=4, pin_memory=True, persistent_workers=True)
    train_loader = DataLoader(PairedDermDataset(train_df, train_tf, train_tf),
                              batch_size=args.batch_size, shuffle=True, drop_last=True, **kw)
    val_loader = DataLoader(PairedDermDataset(val_df, eval_tf, eval_tf), batch_size=32, shuffle=False, **kw)
    test_loader = DataLoader(PairedDermDataset(test_df, eval_tf, eval_tf), batch_size=32, shuffle=False, **kw)

    model = Net(args.mode, args.derm_init).to(device)
    n_pos = train_df["is_melanoma"].sum()
    pos_weight = torch.tensor((len(train_df) - n_pos) / n_pos, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda")

    best_auc, best_state, history = -1.0, None, []
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
        history.append({"epoch": epoch, "train_loss": total / len(train_loader), "val_auc": val_auc})
        print(f"  seed {seed} epoch {epoch}/{args.epochs}  loss {total / len(train_loader):.4f}  val_AUC {val_auc:.4f}")
        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    val_probs, val_labels = predict(model, val_loader, device)
    fpr, tpr, thr = roc_curve(val_labels, val_probs)
    threshold = float(thr[np.argmax(tpr - fpr)])

    test_probs, test_labels = predict(model, test_loader, device)
    test_auc = float(roc_auc_score(test_labels, test_probs))
    sens, spec = sens_spec(test_probs, test_labels, threshold)
    print(f"seed {seed}: TEST AUC {test_auc:.4f}  sens {sens:.3f}  spec {spec:.3f}")
    return {"seed": seed, "val_auc_best": best_auc, "threshold_from_val": threshold,
            "test_auc": test_auc, "test_sensitivity": sens, "test_specificity": spec, "history": history, "val_probs": val_probs.tolist(), "test_probs": test_probs.tolist()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["derm", "clinic", "both"], required=True)
    parser.add_argument("--derm_init", choices=["imagenet", "ham"], default="imagenet")
    parser.add_argument("--epochs", type=int, default=20)
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
    dfs = load_derm7pt_splits()

    print(f"Mode: {args.mode}  derm_init: {args.derm_init}  seeds: {args.seeds}")
    runs = [run_seed(args, s, device, dfs, train_tf, eval_tf) for s in args.seeds]

    summary = {}
    for key in ["val_auc_best", "test_auc", "test_sensitivity", "test_specificity"]:
        vals = np.array([r[key] for r in runs])
        summary[key] = {"mean": float(vals.mean()), "std": float(vals.std())}
    print(f"\nSUMMARY ({args.mode}, derm_init={args.derm_init}, {len(runs)} seeds)")
    for key, v in summary.items():
        print(f"  {key}: {v['mean']:.3f} +/- {v['std']:.3f}")

    tag = args.mode if args.mode == "clinic" else f"{args.mode}_{args.derm_init}"
    with open(OUT_DIR / f"{tag}.json", "w") as f:
        json.dump({"args": vars(args), "summary": summary, "runs": runs}, f, indent=2)


if __name__ == "__main__":
    main()