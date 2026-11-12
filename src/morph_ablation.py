from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from cv_stats import make_fold_indices
from milk10k_data import load_milk10k_splits

ROOT = Path(__file__).resolve().parents[1]
MORPH_CSV = ROOT / "results" / "morph" / "milk10k_morph_features.csv"

SHAPE = ["circularity", "solidity", "elongation", "asymmetry", "perimeter_ratio", "radial_cv"]
FRAMING = ["area_frac", "touches_border"]
COLOUR = ["mean_L", "mean_a", "mean_b", "std_L", "std_a", "std_b", "contrast",
          "texture_entropy", "texture_grad", "edge_sharpness"]


def columns(names):
    return [p + n for p in ("clin_", "derm_") for n in names]


SETS = {
    "framing only (lesion size in the frame, touches the border)": columns(FRAMING),
    "shape only (outline measurements)": columns(SHAPE),
    "colour and texture only": columns(COLOUR),
    "everything": columns(FRAMING + SHAPE + COLOUR),
}


def pooled_auc(df, cols):
    X = df[cols].to_numpy(dtype=np.float32)
    y = df["is_melanoma"].to_numpy()
    probs = np.zeros(len(df))
    for _, tr, _, te in make_fold_indices(df["dx"].to_numpy()):
        clf = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, max_leaf_nodes=15,
                                             class_weight="balanced", random_state=0)
        clf.fit(X[tr], y[tr])
        probs[te] = clf.predict_proba(X[te])[:, 1]
    return roc_auc_score(y, probs)


def main():
    tr, va, te = load_milk10k_splits()
    table = pd.concat([tr, va, te], ignore_index=True)
    table = table.merge(pd.read_csv(MORPH_CSV), on="lesion_id", how="left", validate="one_to_one")

    tasks = {
        "melanoma vs everything else": table,
        "melanoma vs ordinary moles only": table[table["dx"].isin(["MEL", "NV"])].reset_index(drop=True),
    }
    print("Pooled 5-fold AUC of the measurements on their own (0.5 = no signal)\n")
    for name, df in tasks.items():
        print(f"{name}: {len(df)} lesions, {int(df['is_melanoma'].sum())} melanoma")
        for label, cols in SETS.items():
            print(f"   {label:62s} {pooled_auc(df, cols):.3f}")
        print()


if __name__ == "__main__":
    main()
