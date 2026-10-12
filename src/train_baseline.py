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

from dataset import HamDataset

ROOT = Path(__file__).resolve().parents[1]
IMG_DIR = ROOT / "data" / "ham10000" / "images"
SPLIT_DIR = ROOT / "splits"
OUT_DIR = ROOT / "results" / "baseline_efficientnet_b0"
SIZE = 224
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs, labels = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(x)
        probs.append(torch.sigmoid(logits.float()).squeeze(1).cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(probs), np.concatenate(labels)


def sens_spec(probs, labels, thr):
    pred = probs >= thr
    tp = int(((pred == 1) & (labels == 1)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    return tp / (tp + fn), tn / (tn + fp)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train_tf = T.Compose([
        T.Resize((SIZE, SIZE)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(20),
        T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])
    eval_tf = T.Compose([T.Resize((SIZE, SIZE)), T.ToTensor(), T.Normalize(MEAN, STD)])

    train_ds = HamDataset(SPLIT_DIR / "ham10000_train.csv", IMG_DIR, train_tf)
    val_ds = HamDataset(SPLIT_DIR / "ham10000_val.csv", IMG_DIR, eval_tf)
    test_ds = HamDataset(SPLIT_DIR / "ham10000_test.csv", IMG_DIR, eval_tf)

    kw = dict(num_workers=4, pin_memory=True, persistent_workers=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True, **kw)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, **kw)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, **kw)

    model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=1).to(device)

    # Melanoma is rare, so mistakes on melanoma images count more in the loss.
    n_pos = train_ds.df["is_melanoma"].sum()
    n_neg = len(train_ds) - n_pos
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(n_neg / n_pos, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda")

    best_auc, history = 0.0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(x).squeeze(1)
            loss = criterion(logits.float(), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += loss.item()
        scheduler.step()

        val_probs, val_labels = predict(model, val_loader, device)
        val_auc = roc_auc_score(val_labels, val_probs)
        history.append({"epoch": epoch, "train_loss": total / len(train_loader), "val_auc": float(val_auc)})
        print(f"Epoch {epoch}/{args.epochs}  train_loss {total / len(train_loader):.4f}  val_AUC {val_auc:.4f}")
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), OUT_DIR / "best.pt")

    # Final evaluation: best checkpoint, threshold chosen on validation only.
    model.load_state_dict(torch.load(OUT_DIR / "best.pt"))
    val_probs, val_labels = predict(model, val_loader, device)
    fpr, tpr, thr = roc_curve(val_labels, val_probs)
    threshold = float(thr[np.argmax(tpr - fpr)])

    test_probs, test_labels = predict(model, test_loader, device)
    test_auc = float(roc_auc_score(test_labels, test_probs))
    sens, spec = sens_spec(test_probs, test_labels, threshold)

    results = {"val_auc_best": float(best_auc), "threshold_from_val": threshold,
               "test_auc": test_auc, "test_sensitivity": sens, "test_specificity": spec,
               "history": history}
    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nTEST  AUC {test_auc:.4f}  sensitivity {sens:.3f}  specificity {spec:.3f}  (threshold {threshold:.3f})")


if __name__ == "__main__":
    main()