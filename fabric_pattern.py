"""Missing-motif check for repeating-print cloth (REVIEW-only hint, never decides REJECT).

Idea: a printed motif grid has two dominant neighbour vectors. A lattice position that has a
motif on both sides along one vector but almost no printed ink itself is a missing motif.
Works on one photo, needs no reference photo, and does not depend on which part of the cloth
is in view. It does not detect wrong-type or rotated motifs (hand-printed cloth varies too much).

Measured on the pattern cloth (synthetic defects, 14 fresh clean photos): see README.
"""
import cv2
import numpy as np

INK_GRAY = 125          # blurred gray below this = printed ink (dark print on cream cloth)
DENS_RATIO = 0.25       # gap if local ink < this fraction of a typical motif's local ink
TOL = 0.42              # a motif within TOL*spacing of a lattice position counts as present


def _ink(img):
    g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (0, 0), 3)
    return (g < INK_GRAY).astype(np.uint8)


def find_motifs(img, px_per_mm=5.0):
    """Centroids of isolated printed motifs (not touching the frame border)."""
    s = (px_per_mm / 5.0) ** 2
    m = cv2.morphologyEx(_ink(img) * 255, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, _, st, cen = cv2.connectedComponentsWithStats(m)
    H, W = m.shape
    out = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if a < 6000 * s or a > 90000 * s or x < 15 or y < 15 or x + w > W - 15 or y + h > H - 15:
            continue
        out.append((x + w / 2.0, y + h / 2.0))
    return np.array(out, float)


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


def find_gaps(img, px_per_mm=5.0):
    """Returns {"n_motifs", "regular", "gaps": [(x_px, y_px), ...]}."""
    c = find_motifs(img, px_per_mm)
    L = lattice(c)
    if L is None:
        return {"n_motifs": len(c), "regular": False, "gaps": []}
    cand = []
    for v in L:
        n = float(np.linalg.norm(v))
        for a in range(len(c)):
            p, q = c[a] + v, c[a] + 2 * v
            if _nearest(c, p) >= TOL * n and _nearest(c, q) < TOL * n:
                cand.append(p)
    uniq = []
    for p in cand:
        if all(np.linalg.norm(p - r) > 0.5 * np.linalg.norm(L[0]) for r in uniq):
            uniq.append(p)
    ink = _ink(img).astype(np.float32)
    R = int(0.3 * np.linalg.norm(L[0]))

    def dens(p):
        m = np.zeros(ink.shape, np.uint8)
        cv2.circle(m, (int(p[0]), int(p[1])), R, 1, -1)
        return float(ink[m > 0].mean())

    typ = float(np.median([dens(q) for q in c]))
    H, W = ink.shape
    gaps = [p for p in uniq if 0 <= p[0] < W and 0 <= p[1] < H and dens(p) < DENS_RATIO * typ]
    return {"n_motifs": len(c), "regular": True, "gaps": [(float(p[0]), float(p[1])) for p in gaps]}


def pattern_findings(img, px_per_mm, zone_fn):
    """Findings in the pipeline's format: type 'Pattern gap', severity Low (=> at most REVIEW)."""
    r = find_gaps(img, px_per_mm)
    H, W = img.shape[:2]
    f = [{"type": "Pattern gap", "severity": "Low", "confidence": None, "area_mm2": 0.0,
          "width_mm": 0.0, "height_mm": 0.0, "aspect": 1.0,
          "x_mm": round(x / px_per_mm, 1), "y_mm": round(y / px_per_mm, 1),
          "zone": zone_fn(x, y, W, H), "px": (int(x), int(y))} for x, y in r["gaps"]]
    return r, f
