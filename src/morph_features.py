from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from mask_sample import find_lesion

ROOT = Path(__file__).resolve().parents[1]
OUT_CSV = ROOT / "results" / "morph" / "milk10k_morph_features.csv"

FEATURES = ["found", "area_frac", "circularity", "solidity", "elongation", "asymmetry",
            "perimeter_ratio", "radial_cv", "touches_border",
            "mean_L", "mean_a", "mean_b", "std_L", "std_a", "std_b",
            "contrast", "texture_entropy", "texture_grad", "edge_sharpness"]


def empty_features():
    d = {k: np.nan for k in FEATURES}
    d["found"] = 0.0
    return d


def features_from_mask(rgb, mask):
    h, w = mask.shape
    m = mask > 0
    area = float(m.sum())
    if area < 50:
        return empty_features()

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = max(cnts, key=cv2.contourArea)
    carea = max(cv2.contourArea(cnt), 1.0)
    per = cv2.arcLength(cnt, True)
    hull = cv2.convexHull(cnt)
    hull_area = max(cv2.contourArea(hull), 1.0)
    hull_per = max(cv2.arcLength(hull, True), 1.0)

    mom = cv2.moments(mask, binaryImage=True)
    cx, cy = mom["m10"] / mom["m00"], mom["m01"] / mom["m00"]
    mu20, mu02, mu11 = mom["mu20"] / mom["m00"], mom["mu02"] / mom["m00"], mom["mu11"] / mom["m00"]
    common = np.sqrt(4 * mu11 ** 2 + (mu20 - mu02) ** 2)
    l1, l2 = (mu20 + mu02 + common) / 2, (mu20 + mu02 - common) / 2
    elongation = 1.0 - np.sqrt(max(l2, 0.0) / (l1 + 1e-9))
    angle = 0.5 * np.degrees(np.arctan2(2 * mu11, mu20 - mu02))

    # Asymmetry: align the main axis, then compare the lesion with its mirror images
    size = int(1.5 * max(h, w))
    rot = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    rot[0, 2] += size / 2 - cx
    rot[1, 2] += size / 2 - cy
    r = cv2.warpAffine(mask, rot, (size, size), flags=cv2.INTER_NEAREST) > 0
    ious = []
    for f in (np.fliplr(r), np.flipud(r)):
        ious.append((r & f).sum() / max((r | f).sum(), 1))
    asymmetry = 1.0 - float(np.mean(ious))

    pts = cnt[:, 0, :].astype(np.float32)
    radii = np.linalg.norm(pts - np.array([cx, cy], dtype=np.float32), axis=1)
    radial_cv = float(radii.std() / (radii.mean() + 1e-9))

    touches = bool(m[0, :].any() or m[-1, :].any() or m[:, 0].any() or m[:, -1].any())

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    inside = lab[m]
    mean_in, std_in = inside.mean(axis=0), inside.std(axis=0)

    k = max(3, int(0.15 * np.sqrt(area)))
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
    ring = (cv2.dilate(mask, ker) > 0) & ~m
    contrast = float(np.linalg.norm(mean_in - lab[ring].mean(axis=0))) if ring.sum() >= 20 else np.nan

    gray = cv2.GaussianBlur(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), (0, 0), 1)
    hist, _ = np.histogram(gray[m], bins=32, range=(0, 255))
    p = hist[hist > 0] / hist.sum()
    entropy = float(-(p * np.log2(p)).sum())
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    band = (cv2.dilate(mask, k2) > 0) & ~(cv2.erode(mask, k2) > 0)

    return {
        "found": 1.0,
        "area_frac": area / (h * w),
        "circularity": float(min(1.0, 4 * np.pi * carea / (per ** 2 + 1e-9))),
        "solidity": float(min(1.0, carea / hull_area)),
        "elongation": float(elongation),
        "asymmetry": asymmetry,
        "perimeter_ratio": float(per / hull_per),
        "radial_cv": radial_cv,
        "touches_border": float(touches),
        "mean_L": float(mean_in[0]), "mean_a": float(mean_in[1]), "mean_b": float(mean_in[2]),
        "std_L": float(std_in[0]), "std_a": float(std_in[1]), "std_b": float(std_in[2]),
        "contrast": contrast,
        "texture_entropy": entropy,
        "texture_grad": float(mag[m].mean()),
        "edge_sharpness": float(mag[band].mean()) if band.any() else np.nan,
    }


def lesion_features(rgb):
    """rgb: uint8 RGB image of any size. Returns a dict with one value per name in FEATURES."""
    h, w = rgb.shape[:2]
    if max(h, w) > 256:
        s = 256 / max(h, w)
        rgb = cv2.resize(rgb, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    mask = find_lesion(rgb)
    if mask is None:
        return empty_features()
    return features_from_mask(rgb, mask)


def main():
    from tqdm import tqdm

    from milk10k_data import load_milk10k_splits

    train, val, test = load_milk10k_splits()
    table = pd.concat([train, val, test], ignore_index=True)[["lesion_id", "clinic", "derm", "is_melanoma", "dx"]]

    rows = []
    for _, r in tqdm(table.iterrows(), total=len(table)):
        row = {"lesion_id": r["lesion_id"]}
        for kind, prefix in [("clinic", "clin_"), ("derm", "derm_")]:
            img = cv2.imread(r[kind])
            f = empty_features() if img is None else lesion_features(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            row.update({prefix + k: v for k, v in f.items()})
        rows.append(row)

    out = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print("\nSaved", OUT_CSV, out.shape)

    for prefix, name in [("clin_", "clinical"), ("derm_", "dermoscopic")]:
        print(f"No lesion found in {(out[prefix + 'found'] == 0).mean() * 100:.1f}% of {name} images; "
              f"lesion touches the image border in {out[prefix + 'touches_border'].mean() * 100:.1f}%")

    check = out.merge(table[["lesion_id", "is_melanoma"]], on="lesion_id")
    cols = ["derm_asymmetry", "derm_circularity", "derm_perimeter_ratio", "derm_contrast", "derm_std_L",
            "clin_asymmetry", "clin_circularity", "clin_contrast", "clin_std_L"]
    print("\nAverage value, melanoma vs other (a sanity check, not a result):")
    print(check.groupby("is_melanoma")[cols].mean().T.round(3))


if __name__ == "__main__":
    main()
