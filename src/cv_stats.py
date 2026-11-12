import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

SEED = 42


def make_fold_indices(dx, n_folds=5, seed=SEED):
    """Yields (fold, train_idx, val_idx, test_idx) as arrays of row positions.
    Every lesion is test data exactly once; a validation slice is cut from the other folds."""
    dx = np.asarray(dx)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold, (rest, test) in enumerate(skf.split(np.zeros(len(dx)), dx)):
        train, val = train_test_split(rest, test_size=0.15, stratify=dx[rest], random_state=seed)
        yield fold, np.sort(train), np.sort(val), np.sort(test)


def prepare_morph(train_m, val_m, test_m):
    """Fill gaps with the training median, then standardise using training statistics only."""
    med = np.nanmedian(train_m, axis=0)
    med = np.where(np.isnan(med), 0.0, med)

    def fill(a):
        return np.where(np.isnan(a), med, a)

    tr, va, te = fill(train_m), fill(val_m), fill(test_m)
    mu, sd = tr.mean(axis=0), tr.std(axis=0) + 1e-6

    def scale(a):
        return np.clip((a - mu) / sd, -5, 5).astype(np.float32)

    return scale(tr), scale(va), scale(te)


def report(out_dir, arms, n_folds=5, n_boot=2000, seed=0):
    out_dir = Path(out_dir)
    frames = {}
    for arm in arms:
        files = [out_dir / f"oof_{arm}_fold{k}.csv" for k in range(n_folds)]
        if not all(f.exists() for f in files):
            print(f"Report skipped: not all {n_folds} folds are finished for '{arm}'.")
            return None
        frames[arm] = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    base = frames[arms[0]].sort_values("lesion_id").reset_index(drop=True)
    y = base["y"].to_numpy()
    probs = {}
    for arm in arms:
        d = frames[arm].sort_values("lesion_id").reset_index(drop=True)
        assert (d["lesion_id"].values == base["lesion_id"].values).all(), "Arms cover different lesions"
        probs[arm] = d["prob"].to_numpy()

    rng = np.random.default_rng(seed)
    n = len(y)
    boots = {arm: [] for arm in arms}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if y[idx].min() == y[idx].max():
            continue
        for arm in arms:
            boots[arm].append(roc_auc_score(y[idx], probs[arm][idx]))

    result = {"n_lesions": int(n), "n_melanoma": int(y.sum()), "arms": {}, "differences": {}}
    for arm in arms:
        fold_aucs = [roc_auc_score(d["y"], d["prob"]) for _, d in frames[arm].groupby("fold")]
        lo, hi = np.percentile(boots[arm], [2.5, 97.5])
        result["arms"][arm] = {
            "pooled_auc": float(roc_auc_score(y, probs[arm])),
            "ci95": [float(lo), float(hi)],
            "fold_auc_mean": float(np.mean(fold_aucs)),
            "fold_auc_std": float(np.std(fold_aucs)),
        }
    pairs = [("both_morph", "both"), ("both_morph", "morph_only"), ("both", "morph_only")]
    for a, b in pairs:
        if a in arms and b in arms:
            diff = np.array(boots[a]) - np.array(boots[b])
            lo, hi = np.percentile(diff, [2.5, 97.5])
            result["differences"][f"{a} minus {b}"] = {
                "mean_diff": float(roc_auc_score(y, probs[a]) - roc_auc_score(y, probs[b])),
                "ci95": [float(lo), float(hi)],
                "excludes_zero": bool(lo > 0 or hi < 0),
            }

    print(f"\n5-FOLD CROSS-VALIDATION: {n} lesions, {int(y.sum())} melanoma, each tested once")
    for arm, r in result["arms"].items():
        print(f"  {arm:11s} pooled AUC {r['pooled_auc']:.3f} (95% range {r['ci95'][0]:.3f} to {r['ci95'][1]:.3f})"
              f"   per-fold mean {r['fold_auc_mean']:.3f} +/- {r['fold_auc_std']:.3f}")
    for name, r in result["differences"].items():
        print(f"  {name}: {r['mean_diff']:+.3f} (95% range {r['ci95'][0]:+.3f} to {r['ci95'][1]:+.3f})"
              f" -> {'clear difference' if r['excludes_zero'] else 'not clearly different'}")

    with open(out_dir / "report.json", "w") as f:
        json.dump(result, f, indent=2)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(result["arms"])
    vals = [result["arms"][a]["pooled_auc"] for a in names]
    errs = [[v - result["arms"][a]["ci95"][0] for a, v in zip(names, vals)],
            [result["arms"][a]["ci95"][1] - v for a, v in zip(names, vals)]]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(names, vals, yerr=errs, capsize=5, color="#4c78a8")
    ax.set_ylim(max(0.5, min(vals) - 0.1), 1.0)
    ax.set_ylabel("AUC (5-fold cross-validation, pooled)")
    ax.set_title("MILK10k: effect of adding lesion measurements")
    plt.tight_layout()
    plt.savefig(out_dir / "auc_by_arm.png", dpi=120)
    plt.close(fig)
    print("Saved", out_dir / "report.json", "and", out_dir / "auc_by_arm.png")
    return result
