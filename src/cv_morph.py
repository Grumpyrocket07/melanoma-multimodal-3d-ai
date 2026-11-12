import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms as T

from cv_stats import make_fold_indices, prepare_morph, report
from derm7pt_data import PairedDermDataset
from milk10k_data import load_milk10k_splits
from train_derm7pt import MEAN, SIZE, STD, Net, make_encoder

ROOT = Path(__file__).resolve().parents[1]
MORPH_CSV = ROOT / "results" / "morph" / "milk10k_morph_features.csv"


class PairedMorphDataset(PairedDermDataset):
    """One item = (clinical image, dermoscopic image, lesion measurements, label)."""

    def __init__(self, df, morph, clinic_tf, derm_tf):
        super().__init__(df, clinic_tf, derm_tf)
        self.morph = morph.astype(np.float32)

    def __getitem__(self, i):
        clinic, derm, label = super().__getitem__(i)
        return clinic, derm, torch.from_numpy(self.morph[i]), label


class ImagesOnly(nn.Module):
    """The existing fused model: clinical + dermoscopic images, measurements ignored."""

    def __init__(self):
        super().__init__()
        self.net = Net("both", "imagenet")

    def forward(self, clinic, derm, morph):
        return self.net(clinic, derm)


class ImagesPlusMorph(nn.Module):
    """Same two image encoders, plus a small network for the lesion measurements."""

    def __init__(self, n_morph):
        super().__init__()
        self.clinic = make_encoder("imagenet")
        self.derm = make_encoder("imagenet")
        dim = self.clinic.num_features + self.derm.num_features
        self.morph = nn.Sequential(nn.Linear(n_morph, 64), nn.ReLU(), nn.Dropout(0.2),
                                   nn.Linear(64, 32), nn.ReLU())
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(dim + 32, 1))

    def forward(self, clinic, derm, morph):
        feats = torch.cat([self.clinic(clinic), self.derm(derm), self.morph(morph)], dim=1)
        return self.head(feats).squeeze(1)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs = []
    for clinic, derm, morph, _ in loader:
        clinic = clinic.to(device, non_blocking=True)
        derm = derm.to(device, non_blocking=True)
        morph = morph.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(clinic, derm, morph)
        probs.append(torch.sigmoid(logits.float()).cpu().numpy())
    return np.concatenate(probs)


def train_net(args, arm, device, dfs, morphs, seed):
    tr_df, va_df, te_df = dfs
    m_tr, m_va, m_te = morphs
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_tf = T.Compose([
        T.Resize((SIZE, SIZE)), T.RandomHorizontalFlip(), T.RandomVerticalFlip(), T.RandomRotation(20),
        T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1), T.ToTensor(), T.Normalize(MEAN, STD),
    ])
    eval_tf = T.Compose([T.Resize((SIZE, SIZE)), T.ToTensor(), T.Normalize(MEAN, STD)])
    kw = dict(num_workers=4, pin_memory=True, persistent_workers=True)
    train_loader = DataLoader(PairedMorphDataset(tr_df, m_tr, train_tf, train_tf),
                              batch_size=args.batch_size, shuffle=True, drop_last=True, **kw)
    val_loader = DataLoader(PairedMorphDataset(va_df, m_va, eval_tf, eval_tf), batch_size=32, shuffle=False, **kw)
    test_loader = DataLoader(PairedMorphDataset(te_df, m_te, eval_tf, eval_tf), batch_size=32, shuffle=False, **kw)

    model = (ImagesOnly() if arm == "both" else ImagesPlusMorph(m_tr.shape[1])).to(device)
    n_pos = tr_df["is_melanoma"].sum()
    pos_weight = torch.tensor((len(tr_df) - n_pos) / n_pos, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda")
    val_y = va_df["is_melanoma"].to_numpy()

    best_auc, best_state = -1.0, None
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for clinic, derm, morph, y in train_loader:
            clinic = clinic.to(device, non_blocking=True)
            derm = derm.to(device, non_blocking=True)
            morph = morph.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(clinic, derm, morph)
            loss = criterion(logits.float(), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += loss.item()
        scheduler.step()
        val_auc = float(roc_auc_score(val_y, predict(model, val_loader, device)))
        print(f"    {arm} epoch {epoch}/{args.epochs}  loss {total / len(train_loader):.4f}  val_AUC {val_auc:.4f}")
        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return predict(model, test_loader, device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", nargs="+", choices=["morph_only", "both", "both_morph"],
                        default=["morph_only", "both", "both_morph"])
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--out_dir", default="morph_cv")
    args = parser.parse_args()

    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    out_dir = ROOT / "results" / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    tr, va, te = load_milk10k_splits()
    table = pd.concat([tr, va, te], ignore_index=True)
    feats = pd.read_csv(MORPH_CSV)
    table = table.merge(feats, on="lesion_id", how="left", validate="one_to_one")
    feat_cols = [c for c in feats.columns if c != "lesion_id"]
    assert not table["clin_found"].isna().any(), "Measurements are missing for some lesions"
    M = table[feat_cols].to_numpy(dtype=np.float32)
    print(f"{len(table)} lesions, {len(feat_cols)} measurements each")

    for fold, tr_i, va_i, te_i in make_fold_indices(table["dx"].to_numpy()):
        if fold not in args.folds:
            continue
        dfs = [table.iloc[i].reset_index(drop=True) for i in (tr_i, va_i, te_i)]
        morphs = prepare_morph(M[tr_i], M[va_i], M[te_i])
        print(f"\nFOLD {fold}: train {len(tr_i)}, val {len(va_i)}, test {len(te_i)} "
              f"({int(dfs[2]['is_melanoma'].sum())} melanoma in test)")
        for arm in args.arms:
            path = out_dir / f"oof_{arm}_fold{fold}.csv"
            if path.exists():
                print(f"  {arm}: already done, skipping")
                continue
            if arm == "morph_only":
                clf = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, max_leaf_nodes=15,
                                                     class_weight="balanced", random_state=0)
                clf.fit(M[tr_i], dfs[0]["is_melanoma"])
                probs = clf.predict_proba(M[te_i])[:, 1]
            else:
                probs = train_net(args, arm, device, dfs, morphs, seed=fold)
            pd.DataFrame({"lesion_id": dfs[2]["lesion_id"], "y": dfs[2]["is_melanoma"], "prob": probs,
                          "fold": fold}).to_csv(path, index=False)
            print(f"  {arm}: fold {fold} test AUC {roc_auc_score(dfs[2]['is_melanoma'], probs):.4f}")

    report(out_dir, args.arms)


if __name__ == "__main__":
    main()
