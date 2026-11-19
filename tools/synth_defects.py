"""Make synthetic pattern defects from normal photos (for training the pattern check).

  python tools/synth_defects.py                      # all pat_good_* in data/photos_corrected
  python tools/synth_defects.py --src data/photos_corrected --out data/synth --per 2
Writes data/synth/<name>_<kind>.png and data/synth/labels.csv  (kind: miss | flip | swap)
  miss = a motif removed and filled with matching background
  flip = a motif rotated 180 degrees in place (wrong orientation)
  swap = a motif replaced by the other motif type (wrong shape)
"""
import argparse, csv, random
from pathlib import Path
import cv2
import numpy as np


def motifs(img):
    g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (0, 0), 3)
    m = (g < 125).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(m)
    H, W = g.shape
    out = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if a < 6000 or a > 90000 or x < 15 or y < 15 or x + w > W - 15 or y + h > H - 15:
            continue
        mask = (lab == i).astype(np.uint8) * 255
        out.append({"box": (x, y, w, h), "mask": mask, "area": int(a), "fill": a / float(w * h)})
    return out


def cloth_hull(ms, shape):
    pts = np.array([[m["box"][0] + m["box"][2] // 2, m["box"][1] + m["box"][3] // 2] for m in ms], np.int32)
    h = np.zeros(shape, np.uint8)
    cv2.fillConvexPoly(h, cv2.convexHull(pts), 255)
    return cv2.erode(h, np.ones((41, 41), np.uint8))            # keep a margin from the cloth edge


def erase(img, m, rng, ms=None):
    """remove motif m: smooth colour inpaint + real crackle/weave texture taken from motif-free cloth"""
    x, y, w, h = m["box"]
    pad = 40
    H, W = img.shape[:2]
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
    bw, bh = x1 - x0, y1 - y0
    hull = cloth_hull(ms, (H, W)) if ms else np.full((H, W), 255, np.uint8)
    if hull[y0:y1, x0:x1].min() == 0:                                # touches the cloth edge: skip
        return None
    filled = np.zeros_like(m["mask"])                                # fill enclosed gaps (spiral centre)
    cnts, _ = cv2.findContours(m["mask"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(filled, cnts, -1, 255, -1)
    filled = cv2.morphologyEx(filled, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (61, 61)))
    mask = cv2.dilate(filled, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (47, 47)))
    # 1) smooth cloth colour: inpaint on a small copy so it stays flat, then scale back up
    s4 = 4
    small = cv2.resize(img[y0:y1, x0:x1], (bw // s4, bh // s4), interpolation=cv2.INTER_AREA)
    msmall = cv2.resize(mask[y0:y1, x0:x1], (bw // s4, bh // s4), interpolation=cv2.INTER_NEAREST)
    msmall = cv2.dilate(msmall, np.ones((3, 3), np.uint8))
    base = cv2.inpaint(small, msmall, 6, cv2.INPAINT_TELEA)
    base = cv2.resize(base, (bw, bh), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    # 2) texture: high-pass of the window with the least motif ink, taken inside the cloth
    ink = np.zeros((H, W), np.uint8)
    for mm in ms or []:
        ink |= cv2.dilate(mm["mask"], np.ones((15, 15), np.uint8))
    best, bp = 1e9, None
    for py in range(0, H - bh, 25):
        for px in range(0, W - bw, 25):
            if hull[py:py + bh, px:px + bw].min() == 0 or (abs(px - x0) < bw and abs(py - y0) < bh):
                continue
            f = ink[py:py + bh, px:px + bw].mean() / 255.0
            if f < best:
                best, bp = f, (px, py)
    if bp is None or best > 0.55:
        return None
    don = img[bp[1]:bp[1] + bh, bp[0]:bp[0] + bw].astype(np.float32)
    hp = don - cv2.GaussianBlur(don, (0, 0), 7)
    inkw = cv2.GaussianBlur(ink[bp[1]:bp[1] + bh, bp[0]:bp[0] + bw].astype(np.float32) / 255.0, (0, 0), 4)[..., None]
    hp = hp * (1.0 - inkw)                                           # no motif edges in the texture
    fill = base + hp
    ringm = (cv2.dilate(mask, np.ones((41, 41), np.uint8)) - mask)[y0:y1, x0:x1] > 0
    ringm &= (ink[y0:y1, x0:x1] == 0)                                 # cloth only, not neighbouring motifs
    inner = mask[y0:y1, x0:x1] > 0
    if ringm.sum() > 100 and inner.sum() > 100:                       # same average tone as the surrounding cloth
        gain = img[y0:y1, x0:x1][ringm].mean(0) / np.maximum(1.0, fill[inner].mean(0))
        fill = fill * gain.clip(0.85, 1.12)
    fill = np.clip(fill, 0, 255)
    alpha = cv2.GaussianBlur(mask[y0:y1, x0:x1].astype(np.float32) / 255.0, (0, 0), 4)[..., None]
    out = img.copy()
    out[y0:y1, x0:x1] = np.clip(img[y0:y1, x0:x1] * (1 - alpha) + fill * alpha, 0, 255).astype(np.uint8)
    return out


def paste(img, m_dst, src_img, m_src, flip=False, ms=None):
    """erase m_dst, then draw motif m_src (cut from src_img) centred where m_dst was"""
    base = erase(img, m_dst, None, ms)
    if base is None:
        return None
    x, y, w, h = m_src["box"]
    pad = 10
    sx0, sy0 = max(0, x - pad), max(0, y - pad)
    patch = src_img[sy0:y + h + pad, sx0:x + w + pad].copy()
    pm = cv2.dilate(m_src["mask"], np.ones((9, 9), np.uint8))[sy0:y + h + pad, sx0:x + w + pad]
    if flip:
        patch = cv2.rotate(patch, cv2.ROTATE_180); pm = cv2.rotate(pm, cv2.ROTATE_180)
    dx, dy, dw, dh = m_dst["box"]
    cx, cy = dx + dw // 2, dy + dh // 2
    ph, pw = patch.shape[:2]
    if cx - pw // 2 < 2 or cy - ph // 2 < 2 or cx + pw // 2 + 2 > img.shape[1] or cy + ph // 2 + 2 > img.shape[0]:
        return None
    pm = cv2.erode(pm, np.ones((5, 5), np.uint8))
    al = cv2.GaussianBlur(pm.astype(np.float32) / 255.0, (0, 0), 2)[..., None]
    y0, x0 = cy - ph // 2, cx - pw // 2
    roi = base[y0:y0 + ph, x0:x0 + pw].astype(np.float32)
    base[y0:y0 + ph, x0:x0 + pw] = np.clip(roi * (1 - al) + patch.astype(np.float32) * al, 0, 255).astype(np.uint8)
    return base


def kind_of(m):
    return "spiral" if m["fill"] < 0.5 else "paisley"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/photos_corrected")
    ap.add_argument("--out", default="data/synth")
    ap.add_argument("--pattern", default="pat_good_*.png")
    ap.add_argument("--per", type=int, default=2, help="defects per kind per photo")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rows = []
    for f in sorted(Path(a.src).glob(a.pattern)):
        img = cv2.imread(str(f))
        if img is None:
            continue
        ms = motifs(img)
        if len(ms) < 6:
            print("skip", f.name, "(found", len(ms), "motifs)"); continue
        for kind in ("miss", "flip", "swap"):
            for k in range(a.per):
                res = None
                for _try in range(12):
                    dst = rng.choice(ms)
                    if kind == "miss":
                        res = erase(img, dst, rng, ms)
                    elif kind == "flip":
                        res = paste(img, dst, img, dst, flip=True, ms=ms)
                    else:
                        other = [m for m in ms if kind_of(m) != kind_of(dst)]
                        res = paste(img, dst, img, rng.choice(other), ms=ms) if other else None
                    if res is not None:
                        break
                if res is None:
                    continue
                name = "%s_%s%d.png" % (f.stem, kind, k + 1)
                cv2.imwrite(str(out / name), res)
                rows.append([name, kind, *dst["box"]])
    with open(out / "labels.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["file", "kind", "x", "y", "w", "h"]); w.writerows(rows)
    print("wrote", len(rows), "synthetic defect images ->", out)


if __name__ == "__main__":
    main()
