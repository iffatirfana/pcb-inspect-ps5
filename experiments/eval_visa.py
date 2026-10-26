"""Evaluate golden-feature difference on VisA PCB1-PCB4 (run on Kaggle, GPU optional).

    python -m experiments.eval_visa --root /kaggle/working/VisA --refs 5
Prints image AUROC and pixel AUROC per board and saves results/visa_golden_diff.csv.
"""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from sklearn.metrics import roc_auc_score

from align.register import Aligner, AlignError
from vision.golden_diff import GoldenDiff


def load_split(root, cat):
    """Returns (train_good, test) where test = [(image_path, mask_path or None)]."""
    split = Path(root) / "split_csv" / "1cls.csv"
    train, test = [], []
    with open(split, newline="") as f:
        for r in csv.DictReader(f):
            if r["object"] != cat:
                continue
            img = Path(root) / r["image"]
            if r["split"] == "train" and r["label"] == "normal":
                train.append(img)
            elif r["split"] == "test":
                test.append((img, Path(root) / r["mask"] if r["label"] == "anomaly" else None))
    return sorted(train), test


def read(path, width):
    img = cv2.imread(str(path))
    h = round(img.shape[0] * width / img.shape[1])
    return cv2.resize(img, (width, h), interpolation=cv2.INTER_AREA)


def evaluate(root, cat, n_refs, width, aligner_on=True):
    train, test = load_split(root, cat)
    golden = read(train[0], width)
    aligner = Aligner(golden) if aligner_on else None

    def prep(img):
        if aligner is None:
            return img, None
        try:
            al, H, _ = aligner.align(img)
            return al, GoldenDiff.valid_mask(H, img.shape, golden.shape)
        except AlignError:
            return img, None

    gd = GoldenDiff()                       # PyTorch backbone, GPU if available
    refs = [golden] + [prep(read(p, width))[0] for p in train[1:n_refs]]
    gd.set_references(refs)

    img_scores, img_labels, pix_scores, pix_labels = [], [], [], []
    for path, mask_path in test:
        img, valid = prep(read(path, width))
        heat = gd.score(img, valid)
        img_scores.append(float(heat.max()))
        img_labels.append(int(mask_path is not None))
        if mask_path is not None:
            m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            m = cv2.resize(m, (golden.shape[1], golden.shape[0]), interpolation=cv2.INTER_NEAREST)
            if aligner is not None:            # bring the mask into golden coordinates too
                try:
                    _, H, _ = aligner.align(read(path, width))
                    m = cv2.warpPerspective(m, H, (golden.shape[1], golden.shape[0]),
                                            flags=cv2.INTER_NEAREST)
                except AlignError:
                    pass
        else:
            m = np.zeros(golden.shape[:2], np.uint8)
        pix_scores.append(heat[::4, ::4].ravel())
        pix_labels.append((m[::4, ::4] > 0).ravel())
    img_auc = roc_auc_score(img_labels, img_scores)
    pix_auc = roc_auc_score(np.concatenate(pix_labels), np.concatenate(pix_scores))
    return {"board": cat, "refs": n_refs, "width": width, "n_test": len(test),
            "image_auroc": round(img_auc, 4), "pixel_auroc": round(pix_auc, 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="folder containing pcb1..pcb4 and split_csv/")
    ap.add_argument("--boards", default="pcb1,pcb2,pcb3,pcb4")
    ap.add_argument("--refs", type=int, default=5, help="good reference images per board")
    ap.add_argument("--width", type=int, default=1024, help="resize width (px)")
    ap.add_argument("--no-align", action="store_true")
    a = ap.parse_args()
    rows = []
    for cat in a.boards.split(","):
        r = evaluate(a.root, cat, a.refs, a.width, not a.no_align)
        print(r, flush=True)
        rows.append(r)
    Path("results").mkdir(exist_ok=True)
    with open("results/visa_golden_diff.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print("mean image AUROC", round(np.mean([r["image_auroc"] for r in rows]), 4),
          "| mean pixel AUROC", round(np.mean([r["pixel_auroc"] for r in rows]), 4))


if __name__ == "__main__":
    main()
