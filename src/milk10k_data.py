from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data" / "milk10k"
IMG_ROOT = BASE / "images" / "MILK10k_Training_Input"
SPLIT_DIR = ROOT / "splits"
SEED = 42
DX_COLS = ["AKIEC", "BCC", "BEN_OTH", "BKL", "DF", "INF", "MAL_OTH", "MEL", "NV", "SCCKA", "VASC"]
KEEP = ["lesion_id", "clinic", "derm", "dx", "is_melanoma", "skin_tone", "sex", "age", "site",
        "thickness", "confirm"]


def build_lesion_table():
    meta = pd.read_csv(BASE / "MILK10k_Training_Metadata.csv")
    sup = pd.read_csv(BASE / "MILK10k_Training_Supplement.csv")
    gt = pd.read_csv(BASE / "MILK10k_Training_GroundTruth.csv")

    clinic = meta[meta["image_type"] == "clinical: close-up"]
    derm = meta[meta["image_type"] == "dermoscopic"]
    print("Clinical rows:", len(clinic), " Dermoscopic rows:", len(derm))
    assert clinic["lesion_id"].is_unique and derm["lesion_id"].is_unique, "A lesion has two images of one type"
    assert sup["isic_id"].is_unique
    clinic = clinic.set_index("lesion_id")
    derm = derm.set_index("lesion_id")

    t = gt[["lesion_id"] + DX_COLS].set_index("lesion_id")
    print("Lesions whose diagnosis columns do not sum to 1:", int((t[DX_COLS].sum(axis=1) != 1).sum()))
    t["dx"] = t[DX_COLS].idxmax(axis=1)
    t["is_melanoma"] = (t["MEL"] == 1).astype(int)
    t["clinic_id"] = clinic["isic_id"]
    t["derm_id"] = derm["isic_id"]
    missing_pair = t["clinic_id"].isna() | t["derm_id"].isna()
    print("Lesions missing one of the two images:", int(missing_pair.sum()))
    t = t[~missing_pair].copy()

    t["skin_tone"] = derm["skin_tone_class"]
    t["sex"] = derm["sex"]
    t["age"] = derm["age_approx"]
    t["site"] = derm["site"]
    sup = sup.set_index("isic_id")
    t["thickness"] = t["derm_id"].map(sup["invasion_thickness_interval"])
    t["confirm"] = t["derm_id"].map(sup["diagnosis_confirm_type"])
    t = t.reset_index()
    t["clinic"] = t["lesion_id"] + "/" + t["clinic_id"] + ".jpg"
    t["derm"] = t["lesion_id"] + "/" + t["derm_id"] + ".jpg"
    return t


def make_splits(t):
    train_val, test = train_test_split(t, test_size=0.15, stratify=t["dx"], random_state=SEED)
    train, val = train_test_split(train_val, test_size=0.15 / 0.85, stratify=train_val["dx"], random_state=SEED)
    return [d.reset_index(drop=True) for d in (train, val, test)]


def load_milk10k_splits():
    """Returns train, val, test DataFrames; 'clinic' and 'derm' hold full image paths."""
    out = []
    for name in ["train", "val", "test"]:
        df = pd.read_csv(SPLIT_DIR / f"milk10k_{name}.csv")
        df["clinic"] = [str(IMG_ROOT / p) for p in df["clinic"]]
        df["derm"] = [str(IMG_ROOT / p) for p in df["derm"]]
        out.append(df)
    return out


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from PIL import Image

    t = build_lesion_table()
    print("Lesions with both images:", len(t))
    missing_files = [p for col in ["clinic", "derm"] for p in t[col] if not (IMG_ROOT / p).exists()]
    print("Missing image files:", len(missing_files))

    train, val, test = make_splits(t)
    SPLIT_DIR.mkdir(exist_ok=True)
    for name, d in [("train", train), ("val", val), ("test", test)]:
        d[KEEP].to_csv(SPLIT_DIR / f"milk10k_{name}.csv", index=False)
        print(f"{name}: {len(d)} lesions, {int(d['is_melanoma'].sum())} melanoma ({d['is_melanoma'].mean() * 100:.1f}%)")

    print("\nDiagnosis counts (all lesions):")
    print(t["dx"].value_counts())
    print("\nSkin tone class (0 = very dark ... 5 = very light):")
    print(t["skin_tone"].value_counts(dropna=False).sort_index())
    print("\nMelanoma thickness (melanoma lesions only):")
    print(t[t["is_melanoma"] == 1]["thickness"].value_counts(dropna=False))
    print("\nHow the diagnosis was confirmed:")
    print(t["confirm"].value_counts(dropna=False))

    ham = list((ROOT / "data" / "ham10000").glob("HAM10000_metadata*"))
    if ham:
        ham_ids = set(pd.read_csv(ham[0])["image_id"])
        overlap = (set(t["clinic"].str.split("/").str[1].str[:-4]) | set(t["derm"].str.split("/").str[1].str[:-4])) & ham_ids
        print("\nMILK10k images that are also in HAM10000:", len(overlap))

    sample = t.sample(3, random_state=0)
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for j, (_, row) in enumerate(sample.iterrows()):
        axes[0, j].imshow(Image.open(IMG_ROOT / row["clinic"]).convert("RGB"))
        axes[0, j].set_title(f"clinical: {row['dx']}", fontsize=9)
        axes[1, j].imshow(Image.open(IMG_ROOT / row["derm"]).convert("RGB"))
        axes[1, j].set_title("dermoscopic", fontsize=9)
    for ax in axes.flat:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(BASE / "sample_pairs.png", dpi=100)
    print("\nSaved sample figure to", BASE / "sample_pairs.png")