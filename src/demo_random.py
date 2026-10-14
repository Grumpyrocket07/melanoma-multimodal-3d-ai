import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import timm
import torch
from PIL import Image
from torchvision import transforms as T

ROOT = Path(__file__).resolve().parents[1]
IMG_DIR = ROOT / "data" / "ham10000" / "images"
SPLIT = ROOT / "splits" / "ham10000_test.csv"
RUN_DIR = ROOT / "results" / "baseline_efficientnet_b0"
OUT = ROOT / "data" / "demo_random.png"
SIZE = 224
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
DX_NAMES = {"mel": "melanoma", "nv": "benign mole", "bkl": "benign keratosis",
            "bcc": "basal cell carcinoma", "akiec": "actinic keratosis",
            "vasc": "vascular lesion", "df": "dermatofibroma"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_mel", type=int, default=4)
    parser.add_argument("--n_other", type=int, default=8)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    rng = random.Random(args.seed)

    threshold = json.load(open(RUN_DIR / "metrics.json"))["threshold_from_val"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=1)
    model.load_state_dict(torch.load(RUN_DIR / "best.pt", map_location="cpu"))
    model.to(device).eval()

    df = pd.read_csv(SPLIT)
    mel = df[df["is_melanoma"] == 1].sample(args.n_mel, random_state=rng.randrange(10**6))
    other = df[df["is_melanoma"] == 0].sample(args.n_other, random_state=rng.randrange(10**6))
    picks = pd.concat([mel, other]).sample(frac=1, random_state=rng.randrange(10**6))

    tf = T.Compose([T.Resize((SIZE, SIZE)), T.ToTensor(), T.Normalize(MEAN, STD)])
    cols = 4
    rows = (len(picks) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4.4 * rows), squeeze=False)
    correct = 0
    print(f"Cutoff used: {threshold:.3f} (score at or above it = flagged as melanoma)\n")
    for ax, (_, row) in zip(axes.flat, picks.iterrows()):
        img = Image.open(IMG_DIR / f"{row['image_id']}.jpg").convert("RGB")
        with torch.no_grad():
            prob = torch.sigmoid(model(tf(img).unsqueeze(0).to(device))).item()
        flagged = prob >= threshold
        truth = bool(row["is_melanoma"])
        ok = flagged == truth
        correct += ok
        name = DX_NAMES.get(row["dx"], row["dx"])
        verdict = "FLAGGED" if flagged else "not flagged"
        print(f"{row['image_id']}  true: {name:22s} score {prob:.2f}  {verdict:12s} {'correct' if ok else 'WRONG'}")
        ax.imshow(img)
        ax.set_title(f"True: {name}\nScore {prob:.2f} -> {verdict}", fontsize=9, color="green" if ok else "red")
        ax.axis("off")
    for ax in list(axes.flat)[len(picks):]:
        ax.axis("off")

    print(f"\n{correct} of {len(picks)} correct on this sample.")
    plt.suptitle(f"Random test cases: {correct}/{len(picks)} correct (green = correct, red = wrong)")
    plt.tight_layout()
    OUT.parent.mkdir(exist_ok=True)
    plt.savefig(OUT, dpi=100)
    print("Saved to", OUT)
    plt.show()


if __name__ == "__main__":
    main()