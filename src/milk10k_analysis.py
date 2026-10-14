import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from milk10k_data import load_milk10k_splits

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "milk10k" / "both_imagenet.json"
OUT = ROOT / "results" / "milk10k" / "analysis_summary.json"
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)

runs = json.load(open(RES))["runs"]
if "test_probs" not in runs[0]:
    raise SystemExit("No saved scores in the results file.")

train, val, test = load_milk10k_splits()
val_s = np.mean([r["val_probs"] for r in runs], axis=0)
test_s = np.mean([r["test_probs"] for r in runs], axis=0)
assert len(val_s) == len(val) and len(test_s) == len(test)
val_y = val["is_melanoma"].to_numpy()
test_y = test["is_melanoma"].to_numpy()
print(f"AUC of the 3-seed average score: val {roc_auc_score(val_y, val_s):.3f}  test {roc_auc_score(test_y, test_s):.3f}")

low_thr = float(np.quantile(val_s[val_y == 1], 0.05))
high_thr = float(np.quantile(val_s[val_y == 0], 0.90))
print(f"Cutoffs from validation: low if score < {low_thr:.3f}, high if score >= {high_thr:.3f}")
if low_thr >= high_thr:
    raise SystemExit("The two cutoffs overlap.")

test = test.copy()
test["score"] = test_s
test["level"] = np.where(test_s < low_thr, "low", np.where(test_s >= high_thr, "high", "medium"))

tab = test.groupby("level").agg(cases=("is_melanoma", "size"), melanomas=("is_melanoma", "sum"))
tab = tab.reindex(["low", "medium", "high"]).fillna(0).astype(int)
tab["melanoma_rate_%"] = (tab["melanomas"] / tab["cases"].clip(lower=1) * 100).round(1)
tab["share_of_all_melanomas_%"] = (tab["melanomas"] / test_y.sum() * 100).round(1)
print(f"\nTEST SET: {len(test)} lesions, {int(test_y.sum())} melanoma ({test_y.mean() * 100:.1f}% overall)")
print(tab)

diag = test.groupby("dx").agg(
    cases=("level", "size"),
    low_pct=("level", lambda x: (x == "low").mean() * 100),
    medium_pct=("level", lambda x: (x == "medium").mean() * 100),
    high_pct=("level", lambda x: (x == "high").mean() * 100),
).round(1).sort_values("high_pct", ascending=False)
print("\nHOW EACH DIAGNOSIS IS RATED (test set, % per level):")
print(diag)

mel = test[test["is_melanoma"] == 1].copy()
mel["thickness"] = mel["thickness"].fillna("not recorded")
thick = mel.groupby("thickness").agg(
    cases=("level", "size"),
    low_pct=("level", lambda x: (x == "low").mean() * 100),
    medium_pct=("level", lambda x: (x == "medium").mean() * 100),
    high_pct=("level", lambda x: (x == "high").mean() * 100),
).round(1)
print("\nMELANOMAS BY THICKNESS (test set):")
print(thick)

tone_map = {0: "0-2 (darker)", 1: "0-2 (darker)", 2: "0-2 (darker)", 3: "3 (middle)", 4: "4-5 (lighter)", 5: "4-5 (lighter)"}
test["tone_group"] = test["skin_tone"].map(tone_map)
rows = []
for g, d in test.groupby("tone_group"):
    n_mel = int(d["is_melanoma"].sum())
    auc = roc_auc_score(d["is_melanoma"], d["score"]) if n_mel >= 5 and n_mel < len(d) - 4 else np.nan
    found = (d[d["is_melanoma"] == 1]["level"] != "low").mean() * 100 if n_mel else np.nan
    rows.append({"skin_tone_group": g, "lesions": len(d), "melanomas": n_mel,
                 "AUC": round(auc, 3) if auc == auc else np.nan,
                 "melanomas_rated_medium_or_high_%": round(found, 1) if found == found else np.nan})
tone = pd.DataFrame(rows).set_index("skin_tone_group")
print("\nBY SKIN TONE (test set; AUC shown only with at least 5 melanomas):")
print(tone)

summary = {"cutoffs": {"low_below": low_thr, "high_at_or_above": high_thr},
           "levels": tab.to_dict(), "per_diagnosis": diag.to_dict(),
           "melanoma_by_thickness": thick.to_dict(), "by_skin_tone": tone.to_dict()}
with open(OUT, "w") as f:
    json.dump(summary, f, indent=2)
print("\nSaved", OUT)