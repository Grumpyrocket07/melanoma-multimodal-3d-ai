from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from milk10k_data import load_milk10k_splits

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "milk10k"


def find_lesion(rgb):
    """Returns a 0/255 mask the size of the image, or None if no lesion-like region is found."""
    h, w = rgb.shape[:2]
    s = 256 / max(h, w)
    small = cv2.resize(rgb, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    sh, sw = small.shape[:2]
    lab = cv2.cvtColor(small, cv2.COLOR_RGB2LAB)
    ch = cv2.GaussianBlur(lab[:, :, 0], (0, 0), 2)
    _, m = cv2.threshold(ch, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)

    n, labels, stats, cents = cv2.connectedComponentsWithStats(m)
    best, best_score = None, 0.0
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < 0.02 * sh * sw:
            continue
        sides = int(x == 0) + int(y == 0) + int(x + bw == sw) + int(y + bh == sh)
        cx, cy = cents[i]
        dist = float(np.hypot(cx / sw - 0.5, cy / sh - 0.5))
        score = area * (1.0 - dist) * (0.1 if sides >= 2 else 1.0)
        if score > best_score:
            best, best_score = i, score
    if best is None:
        return None

    mask = (labels == best).astype(np.uint8) * 255
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, cnts, -1, 255, thickness=cv2.FILLED)
    return cv2.resize(filled, (w, h), interpolation=cv2.INTER_NEAREST)


def main():
    train, _, _ = load_milk10k_splits()
    sample = train.sample(24, random_state=1)
    for kind in ["clinic", "derm"]:
        fig, axes = plt.subplots(4, 6, figsize=(18, 12))
        missing = 0
        for ax, (_, row) in zip(axes.flat, sample.iterrows()):
            img = cv2.cvtColor(cv2.imread(row[kind]), cv2.COLOR_BGR2RGB)
            mask = find_lesion(img)
            if mask is None:
                missing += 1
            else:
                cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                img = img.copy()
                cv2.drawContours(img, cnts, -1, (0, 255, 0), max(2, img.shape[1] // 150))
            ax.imshow(img)
            ax.set_title(row["dx"], fontsize=9)
            ax.axis("off")
        plt.tight_layout()
        out = OUT / f"mask_check_{kind}.png"
        plt.savefig(out, dpi=70)
        plt.close(fig)
        print(f"{kind}: no lesion found in {missing} of {len(sample)}; saved {out}")


if __name__ == "__main__":
    main()