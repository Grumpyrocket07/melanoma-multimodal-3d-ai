import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "results" / "milk10k_cv"
MODES = ["derm", "clinic", "both"]
N_BOOT = 2000

runs = {m: [json.load(open(D / f"fold{f}_{m}.json")) for f in range(5)] for m in MODES}

ids = {m: sum((r["lesion_ids"] for r in runs[m]), []) for m in MODES}
assert ids["derm"] == ids["clinic"] == ids["both"], "Lesion order differs between modes"
y = np.array(sum((r["labels"] for r in runs["derm"]), []))
probs = {m: np.array(sum((r["test_probs"] for r in runs[m]), [])) for m in MODES}
n = len(y)
print(f"Pooled out-of-fold lesions: {n}  melanoma: {int(y.sum())}\n")

summary = {}
for m in MODES:
    per_fold = np.array([r["test_auc"] for r in runs[m]])
    pooled = float(roc_auc_score(y, probs[m]))
    summary[m] = {"pooled_auc": pooled, "per_fold_auc": per_fold.tolist(),
                  "per_fold_mean": float(per_fold.mean()), "per_fold_std": float(per_fold.std())}
    print(f"{m:7s} pooled AUC {pooled:.3f} | per-fold mean {per_fold.mean():.3f} +/- {per_fold.std():.3f} "
          f"| folds: {', '.join(f'{a:.3f}' for a in per_fold)}")

rng = np.random.default_rng(0)
d_derm, d_clin = [], []
for _ in range(N_BOOT):
    idx = rng.integers(0, n, n)
    yb = y[idx]
    if yb.sum() == 0 or yb.sum() == n:
        continue
    a_both = roc_auc_score(yb, probs["both"][idx])
    d_derm.append(a_both - roc_auc_score(yb, probs["derm"][idx]))
    d_clin.append(a_both - roc_auc_score(yb, probs["clinic"][idx]))

print("\nPaired bootstrap, 95% CI (resampling lesions, 2000 times):")
for name, d, obs in [("both - derm  ", d_derm, summary["both"]["pooled_auc"] - summary["derm"]["pooled_auc"]),
                     ("both - clinic", d_clin, summary["both"]["pooled_auc"] - summary["clinic"]["pooled_auc"])]:
    lo, hi = np.percentile(d, [2.5, 97.5])
    verdict = "CI excludes 0" if (lo > 0 or hi < 0) else "CI includes 0"
    print(f"  AUC({name}) = {obs:+.3f}   95% CI [{lo:+.3f}, {hi:+.3f}]   {verdict}")
    summary[f"diff_{name.strip().replace(' ', '')}"] = {"observed": float(obs), "ci_low": float(lo), "ci_high": float(hi)}

with open(D / "cv_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("\nSaved", D / "cv_summary.json")