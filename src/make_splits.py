from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "ham10000"
OUT_DIR = ROOT / "splits"
SEED = 42


def main():
    meta_path = next(DATA_DIR.glob("HAM10000_metadata*"))
    df = pd.read_csv(meta_path)
    df["is_melanoma"] = (df["dx"] == "mel").astype(int)

    # 7 folds, grouped by lesion so one lesion never appears in two folds,
    # and stratified by diagnosis so each fold has a similar class mix.
    sgkf = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=SEED)
    df["fold"] = -1
    for fold, (_, idx) in enumerate(sgkf.split(df, y=df["dx"], groups=df["lesion_id"])):
        df.loc[idx, "fold"] = fold

    test = df[df["fold"] == 0].drop(columns="fold")
    val = df[df["fold"] == 1].drop(columns="fold")
    train = df[df["fold"] >= 2].drop(columns="fold")

    # Safety check: no lesion may appear in more than one split.
    for name, a, b in [("train/val", train, val), ("train/test", train, test), ("val/test", val, test)]:
        overlap = set(a["lesion_id"]) & set(b["lesion_id"])
        assert not overlap, f"Lesion overlap between {name}: {len(overlap)}"

    OUT_DIR.mkdir(exist_ok=True)
    train.to_csv(OUT_DIR / "ham10000_train.csv", index=False)
    val.to_csv(OUT_DIR / "ham10000_val.csv", index=False)
    test.to_csv(OUT_DIR / "ham10000_test.csv", index=False)

    print("Split sizes:", len(train), len(val), len(test))
    print("\nDiagnosis counts per split:")
    print(pd.DataFrame({"train": train["dx"].value_counts(),
                        "val": val["dx"].value_counts(),
                        "test": test["dx"].value_counts()}))
    print("\nMelanoma share per split (%):",
          round(train["is_melanoma"].mean() * 100, 1),
          round(val["is_melanoma"].mean() * 100, 1),
          round(test["is_melanoma"].mean() * 100, 1))
    print("\nNo lesion overlap between splits.")


if __name__ == "__main__":
    main()