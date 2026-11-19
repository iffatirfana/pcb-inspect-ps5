"""Pattern checks for repeating-print cloth. REVIEW-only hints: they never decide REWORK or REJECT.

1. find_gaps(): a missing printed motif. A printed motif grid has two dominant neighbour vectors. A lattice
   position that has a motif on both sides along one vector but almost no printed ink itself is a missing
   motif. Works on one photo, does not depend on which part of the cloth is in view. Wrong-type or flipped
   motifs are not detected (hand-printed cloth varies too much).
2. build_profile(): learns the typical motif spacing and ink density from good reference photos, so the
   check can warn when a test photo is at a different scale than the references.
3. compare_to_reference(): registers the test photo to a reference photo of the same part of the cloth and
   lists motifs whose shape differs. Information only (not used for the verdict): it is only meaningful when
   both photos show the same area of the cloth.

Measured numbers are in README.md.
"""
import json
from pathlib import Path

import cv2
import numpy as np

INK_GRAY = 125          # blurred gray below this = printed ink (dark print on cream cloth)
DENS_RATIO = 0.25       # gap if local ink < this fraction of a typical motif's local ink
TOL = 0.42              # a motif within TOL*spacing of a lattice position counts as present
REF_DIR = Path("data/pattern_refs")
IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp")


def _ink(img):
    g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (0, 0), 3)
    return (g < INK_GRAY).astype(np.uint8)


def _components(img, px_per_mm=5.0):
    s = (px_per_mm / 5.0) ** 2
    m = cv2.morphologyEx(_ink(img) * 255, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    H, W = m.shape
    out = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if a < 6000 * s or a > 90000 * s or x < 15 or y < 15 or x + w > W - 15 or y + h > H - 15:
            continue
        out.append((int(x), int(y), int(w), int(h), lab == i))
    return out


def find_motifs(img, px_per_mm=5.0):
    """Centroids of isolated printed motifs (not touching the frame border)."""
    return np.array([(x + w / 2.0, y + h / 2.0) for x, y, w, h, _ in _components(img, px_per_mm)], float)


def _nearest(c, p):
    return float(np.min(np.linalg.norm(c - p, axis=1)))


def lattice(c):
    """Two dominant neighbour vectors (px), or None when the print is not a regular grid."""
    if len(c) < 12:
        return None
    d = np.linalg.norm(c[:, None] - c[None], axis=2)
    idx = np.argsort(d, axis=1)[:, :5]
    V = []
    for a in range(len(c)):
        for b in idx[a][1:]:
            v = c[b] - c[a]
            if v[0] < 0 or (v[0] == 0 and v[1] < 0):
                v = -v
            V.append(v)
    V = np.array(V)
    ang = np.degrees(np.arctan2(V[:, 1], V[:, 0])) % 180
    h, _ = np.histogram(ang, bins=36, range=(0, 180))
    p1 = int(np.argmax(h))
    rest = h.copy()
    for k in range(-4, 6):
        rest[(p1 + k) % 36] = 0
    p2 = int(np.argmax(rest))

    def near(p):
        return np.abs(((ang - (p * 5 + 2.5)) + 90) % 180 - 90) < 12
    m1, m2 = near(p1), near(p2)
    if m1.sum() < 8 or m2.sum() < 8:
        return None
    return np.median(V[m1], 0), np.median(V[m2], 0)


def _density_fn(img, R):
    ink = _ink(img).astype(np.float32)

    def dens(p):
        m = np.zeros(ink.shape, np.uint8)
        cv2.circle(m, (int(p[0]), int(p[1])), R, 1, -1)
        return float(ink[m > 0].mean())
    return dens, ink.shape


def find_gaps(img, px_per_mm=5.0, profile=None):
    """Returns {"n_motifs", "regular", "gaps": [(x_px, y_px), ...], "spacing_ratio"}.
    spacing_ratio = this photo's motif spacing / the references' spacing (about 1.0 when the scale matches)."""
    c = find_motifs(img, px_per_mm)
    L = lattice(c)
    if L is None:
        return {"n_motifs": len(c), "regular": False, "gaps": [], "spacing_ratio": None}
    sp = float(np.linalg.norm(L[0]))
    cand = []
    for v in L:
        n = float(np.linalg.norm(v))
        for a in range(len(c)):
            p, q = c[a] + v, c[a] + 2 * v
            if _nearest(c, p) >= TOL * n and _nearest(c, q) < TOL * n:
                cand.append(p)
    uniq = []
    for p in cand:
        if all(np.linalg.norm(p - r) > 0.5 * sp for r in uniq):
            uniq.append(p)
    dens, (H, W) = _density_fn(img, int(0.3 * sp))
    typ = float(np.median([dens(q) for q in c]))
    gaps = [p for p in uniq if 0 <= p[0] < W and 0 <= p[1] < H and dens(p) < DENS_RATIO * typ]
    ratio = round(sp / profile["spacing"], 2) if profile else None
    return {"n_motifs": len(c), "regular": True, "gaps": [(float(p[0]), float(p[1])) for p in gaps],
            "spacing_ratio": ratio}


# ------------------------------------------------------------------ reference photos

def ref_dir(fabric):
    return REF_DIR / fabric


def list_refs(fabric):
    d = ref_dir(fabric)
    return sorted(p for p in d.glob("*") if p.suffix.lower() in IMG_EXT) if d.exists() else []


def add_ref(fabric, img, name=None):
    """Save a good reference photo (BGR array). Returns the file name."""
    d = ref_dir(fabric)
    d.mkdir(parents=True, exist_ok=True)
    n = 1 + max([int(p.stem.split("_")[-1]) for p in list_refs(fabric) if p.stem.split("_")[-1].isdigit()] or [0])
    fn = d / ("ref_%02d.png" % n)
    cv2.imwrite(str(fn), img)
    _PROFILE_CACHE.pop(fabric, None)
    return fn.name


def delete_ref(fabric, name):
    p = ref_dir(fabric) / Path(name).name
    if p.exists() and p.suffix.lower() in IMG_EXT:
        p.unlink()
        _PROFILE_CACHE.pop(fabric, None)
        return True
    return False


_PROFILE_CACHE = {}


def load_profile(fabric, px_per_mm=5.0):
    """Profile of the good reference photos, or None when there are none. Cached until refs change."""
    refs = list_refs(fabric)
    key = tuple((p.name, p.stat().st_mtime) for p in refs)
    hit = _PROFILE_CACHE.get(fabric)
    if hit and hit[0] == key:
        return hit[1]
    prof = build_profile([cv2.imread(str(p)) for p in refs], px_per_mm) if refs else None
    _PROFILE_CACHE[fabric] = (key, prof)
    return prof


def build_profile(images, px_per_mm=5.0):
    """Typical motif spacing (px) and motif count from good photos. None if no photo shows a regular grid."""
    sps, counts = [], []
    for im in images:
        if im is None:
            continue
        c = find_motifs(im, px_per_mm)
        L = lattice(c)
        if L is not None:
            sps.append(float(np.linalg.norm(L[0])))
            counts.append(len(c))
    if not sps:
        return None
    return {"n_refs": len(sps), "spacing": float(np.median(sps)), "motifs": float(np.median(counts))}


# ------------------------------------------------------------------ compare with a reference photo

def _register(test, ref, nfeat=4000):
    """Homography test -> ref (ORB + RANSAC; copes with rotation). Returns (H, inliers, ratio) or None."""
    orb = cv2.ORB_create(nfeat)
    g1, g2 = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY), cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    k1, d1 = orb.detectAndCompute(g1, None)
    k2, d2 = orb.detectAndCompute(g2, None)
    if d1 is None or d2 is None or len(k1) < 20 or len(k2) < 20:
        return None
    good = []
    for pair in cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(d1, d2, k=2):
        if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance:
            good.append(pair[0])
    if len(good) < 12:
        return None
    src = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
    if H is None:
        return None
    n = int(mask.sum())
    return H, n, n / float(len(good))


def _local_iou(ref_ink, test_ink, box, mask, search=24):
    x, y, w, h = box
    R = mask[y:y + h, x:x + w].astype(np.float32)
    y0, x0 = max(0, y - search), max(0, x - search)
    y1, x1 = min(test_ink.shape[0], y + h + search), min(test_ink.shape[1], x + w + search)
    T = test_ink[y0:y1, x0:x1]
    if T.shape[0] < h or T.shape[1] < w:
        return None
    ov = cv2.matchTemplate(T, R, cv2.TM_CCORR)
    ts = cv2.matchTemplate(T, np.ones_like(R), cv2.TM_CCORR)
    return float((ov / np.maximum(1.0, R.sum() + ts - ov)).max())


def compare_to_reference(test, refs, px_per_mm=5.0, min_iou=0.4):
    """Compare a test photo with the best-matching reference photo (BGR arrays, a list or one).
    Returns {"matched", "inliers", "ratio", "ref_index", "checked", "diffs": [(x, y, w, h, iou)] in reference
    pixels, "warped": test photo in the reference frame, "ref": the reference}.
    matched is False when the photos do not show the same part of the cloth; then no differences are listed."""
    refs = refs if isinstance(refs, (list, tuple)) else [refs]
    best = None
    for i, r in enumerate(refs):
        reg = _register(test, r)
        if reg and (best is None or reg[1] > best[1][1]):
            best = (i, reg)
    if best is None:
        return {"matched": False, "inliers": 0, "ratio": 0.0, "ref_index": None, "checked": 0, "diffs": [],
                "warped": None, "ref": None}
    i, (H, n, ratio) = best
    ref = refs[i]
    out = {"matched": n >= 40 and ratio >= 0.35, "inliers": n, "ratio": round(ratio, 2), "ref_index": i,
           "checked": 0, "diffs": [], "warped": None, "ref": ref, "H": H}
    hh, ww = ref.shape[:2]
    warped = cv2.warpPerspective(test, H, (ww, hh))
    valid = cv2.erode(cv2.warpPerspective(np.full(test.shape[:2], 255, np.uint8), H, (ww, hh)), np.ones((61, 61), np.uint8)) > 0
    out["warped"] = warped
    if not out["matched"]:
        return out
    rink, tink = _ink(ref).astype(np.float32), _ink(warped).astype(np.float32)
    for x, y, w, h, mask in _components(ref, px_per_mm):
        if not valid[y:y + h, x:x + w].all():
            continue
        iou = _local_iou(rink, tink, (x, y, w, h), mask)
        if iou is None:
            continue
        out["checked"] += 1
        if iou < min_iou:
            out["diffs"].append((x, y, w, h, round(iou, 2)))
    return out


# ------------------------------------------------------------------ pipeline findings

def pattern_findings(img, px_per_mm, zone_fn, profile=None):
    """Findings in the pipeline's format: type 'Pattern gap', severity Low (=> at most REVIEW)."""
    r = find_gaps(img, px_per_mm, profile)
    H, W = img.shape[:2]
    f = [{"type": "Pattern gap", "severity": "Low", "confidence": None, "area_mm2": 0.0,
          "width_mm": 0.0, "height_mm": 0.0, "aspect": 1.0,
          "x_mm": round(x / px_per_mm, 1), "y_mm": round(y / px_per_mm, 1),
          "zone": zone_fn(x, y, W, H), "px": (int(x), int(y))} for x, y in r["gaps"]]
    return r, f
