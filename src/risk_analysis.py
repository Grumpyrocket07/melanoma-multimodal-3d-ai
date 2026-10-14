import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from derm7pt_data import load_derm7pt_splits

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "derm7pt" / "both_imagenet.json"
OUT = ROOT / "results" / "derm7pt" / "risk_summary.json"
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)

runs = json.load(open(RES))["runs"]
if "test_probs" not in runs[0]:
    raise SystemExit("No saved scores found. Make the edit in train_derm7pt.py and re-run: python src\\train_derm7pt.py --mode both")

train, val, test = load_derm7pt_splits()
# One score per case: the average over the 3 seeds
val_s = np.mean([r["val_probs"] for r in runs], axis=0)
test_s = np.mean([r["test_probs"] for r in runs], axis=0)
assert len(val_s) == len(val) and len(test_s) == len(test)
val_y = val["is_melanoma"].to_numpy()
test_y = test["is_melanoma"].to_numpy()

print(f"AUC of the 3-seed average score: val {roc_auc_score(val_y, val_s):.3f}  test {roc_auc_score(test_y, test_s):.3f}")

# Cutoffs come from the validation set only
low_thr = float(np.quantile(val_s[val_y == 1], 0.05))
high_thr = float(np.quantile(val_s[val_y == 0], 0.90))
print(f"Cutoffs from validation: low if score < {low_thr:.3f}, high if score >= {high_thr:.3f}")
if low_thr >= high_thr:
    raise SystemExit("The two cutoffs overlap, so three levels are not possible with this model.")


def assign(scores):
    return np.where(scores < low_thr, "low", np.where(scores >= high_thr, "high", "medium"))


test = test.copy()
test["score"] = test_s
test["level"] = assign(test_s)

tab = test.groupby("level").agg(cases=("is_melanoma", "size"), melanomas=("is_melanoma", "sum"))
tab = tab.reindex(["low", "medium", "high"]).fillna(0).astype(int)
tab["melanoma_rate_%"] = (tab["melanomas"] / tab["cases"].clip(lower=1) * 100).round(1)
tab["share_of_all_melanomas_%"] = (tab["melanomas"] / test_y.sum() * 100).round(1)
print(f"\nTEST SET: {len(test)} cases, {int(test_y.sum())} melanoma ({test_y.mean() * 100:.1f}% overall)")
print(tab)

diag = test.groupby("diagnosis").agg(
    cases=("level", "size"),
    mean_score=("score", "mean"),
    low_pct=("level", lambda x: (x == "low").mean() * 100),
    medium_pct=("level", lambda x: (x == "medium").mean() * 100),
    high_pct=("level", lambda x: (x == "high").mean() * 100),
).round(2).sort_values("mean_score", ascending=False)
print("\nHOW EACH DIAGNOSIS IS RATED (test set, % of its cases per level):")
print(diag)

early = test[test["diagnosis"].isin(["melanoma (in situ)", "melanoma (less than 0.76 mm)"])]
print(f"\nEARLY MELANOMAS (in situ + thinner than 0.76 mm): {len(early)} cases")
print(f"  rated medium or high: {(early['level'] != 'low').mean() * 100:.1f}%")
print(f"  rated high: {(early['level'] == 'high').mean() * 100:.1f}%")

summary = {
    "cutoffs": {"low_below": low_thr, "high_at_or_above": high_thr},
    "levels": tab.to_dict(),
    "per_diagnosis": diag.to_dict(),
    "early_melanoma": {"cases": int(len(early)),
                       "pct_medium_or_high": float((early["level"] != "low").mean() * 100),
                       "pct_high": float((early["level"] == "high").mean() * 100)},
}
with open(OUT, "w") as f:
    json.dump(summary, f, indent=2)
print("\nSaved", OUT)