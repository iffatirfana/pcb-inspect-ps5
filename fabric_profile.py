"""Fabric profile: heatmap -> findings (mm units) -> severity -> verdict -> LLM payload.

All thresholds in DEFAULT_RULES are PLACEHOLDERS. Tune them on the validation
photos once real heatmaps exist; do not present them as validated.
"""
import cv2
import numpy as np

DEFAULT_RULES = {
    "min_area_mm2": 0.5,       # ignore specks smaller than this
    "low_max_mm2": 2.0,        # area < 2 mm2  -> Low
    "medium_max_mm2": 10.0,    # 2..10 mm2     -> Medium, above -> High
    "max_defects_for_review": 3,  # more Low findings than this -> REWORK
}


def px_per_mm_from_ruler(pixel_distance, mm_distance):
    """Two points on a ruler photo: px distance between them and their real mm."""
    if pixel_distance <= 0 or mm_distance <= 0:
        raise ValueError("distances must be positive")
    return pixel_distance / mm_distance


def crop_image(img, crop):
    """crop = (x, y, w, h) in pixels, or None."""
    if crop is None:
        return img
    x, y, w, h = crop
    return img[y:y + h, x:x + w]


def zone_3x3(cx_px, cy_px, width_px, height_px):
    cols = ["left", "centre", "right"]
    rows = ["top", "middle", "bottom"]
    c = min(2, int(3 * cx_px / width_px))
    r = min(2, int(3 * cy_px / height_px))
    return "%s-%s" % (rows[r], cols[c])


def severity_from_area(area_mm2, rules=DEFAULT_RULES):
    if area_mm2 < rules["low_max_mm2"]:
        return "Low"
    if area_mm2 < rules["medium_max_mm2"]:
        return "Medium"
    return "High"


def findings_from_heatmap(heatmap, threshold, px_per_mm, rules=DEFAULT_RULES):
    """heatmap: 2D float array, same pixel grid as the cropped photo.
    Returns list of findings sorted by area, largest first."""
    h, w = heatmap.shape
    mask = (heatmap > threshold).astype(np.uint8)
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
    px2_per_mm2 = px_per_mm ** 2
    out = []
    for i in range(1, n):
        area_px = int(stats[i, cv2.CC_STAT_AREA])
        area_mm2 = area_px / px2_per_mm2
        if area_mm2 < rules["min_area_mm2"]:
            continue
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]
        cx, cy = cents[i]
        region = heatmap[labels == i]
        out.append({
            "area_mm2": round(area_mm2, 2),
            "width_mm": round(bw / px_per_mm, 1),
            "height_mm": round(bh / px_per_mm, 1),
            "aspect": round(max(bw, bh) / max(1, min(bw, bh)), 1),
            "x_mm": round(cx / px_per_mm, 1),
            "y_mm": round(cy / px_per_mm, 1),
            "zone": zone_3x3(cx, cy, w, h),
            "peak_score": round(float(region.max()), 3),
            "severity": severity_from_area(area_mm2, rules),
        })
    out.sort(key=lambda f: f["area_mm2"], reverse=True)
    return out


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def findings_from_detections(dets, img_shape, px_per_mm, rules=DEFAULT_RULES, merge_iou=0.3):
    """dets: list of {"type", "confidence", "box": (x1,y1,x2,y2), "polygon": Nx2 array or None}
    in the pixel grid of the cropped photo (see fabric_yolo.YoloDetector).
    Overlapping boxes (any class) are merged, keeping the most confident one.
    Returns findings in the same schema as findings_from_heatmap, plus "type" and "confidence"."""
    h, w = img_shape[:2]
    kept = []
    for d in sorted(dets, key=lambda d: d["confidence"], reverse=True):
        if all(_iou(d["box"], k["box"]) < merge_iou for k in kept):
            kept.append(d)
    out = []
    for d in kept:
        x1, y1, x2, y2 = d["box"]
        bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
        poly = d.get("polygon")
        if poly is not None and len(poly) >= 3:
            cnt = np.asarray(poly, np.float32).reshape(-1, 1, 2)
            area_px = float(cv2.contourArea(cnt))
            m = cv2.moments(cnt)
            cx, cy = ((m["m10"] / m["m00"], m["m01"] / m["m00"]) if m["m00"] > 0
                      else ((x1 + x2) / 2, (y1 + y2) / 2))
        else:
            area_px, cx, cy = bw * bh, (x1 + x2) / 2, (y1 + y2) / 2
        area_mm2 = area_px / (px_per_mm ** 2)
        if area_mm2 < rules["min_area_mm2"]:
            continue
        out.append({
            "type": d["type"],
            "confidence": round(float(d["confidence"]), 2),
            "area_mm2": round(area_mm2, 2),
            "width_mm": round(bw / px_per_mm, 1),
            "height_mm": round(bh / px_per_mm, 1),
            "aspect": round(max(bw, bh) / max(1.0, min(bw, bh)), 1),
            "x_mm": round(cx / px_per_mm, 1),
            "y_mm": round(cy / px_per_mm, 1),
            "zone": zone_3x3(cx, cy, w, h),
            "peak_score": round(float(d["confidence"]), 3),
            "severity": severity_from_area(area_mm2, rules),
        })
    out.sort(key=lambda f: f["area_mm2"], reverse=True)
    return out


def verdict_from_findings(findings, rules=DEFAULT_RULES):
    """PASS / REVIEW / REWORK / REJECT. Rule-based, never decided by the LLM."""
    if not findings:
        return "PASS"
    sev = [f["severity"] for f in findings]
    if "High" in sev:
        return "REJECT"
    if "Medium" in sev:
        return "REWORK"
    if len(findings) > rules["max_defects_for_review"]:
        return "REWORK"
    return "REVIEW"


def build_llm_payload(findings, verdict, fabric="white cotton", sop_text=""):
    """Structured input for the advice step. Verdict is fixed here and must
    be repeated, not changed, by the model."""
    lines = ["Fabric: %s" % fabric,
             "Verdict (fixed, do not change): %s" % verdict,
             "Findings: %d" % len(findings)]
    for i, f in enumerate(findings, 1):
        kind = ("%s, " % f["type"]) if f.get("type") else ""
        lines.append("%d) %s%s severity, %.1f mm2 (%.1f x %.1f mm), at %s (x=%.1f, y=%.1f mm)"
                     % (i, kind, f["severity"], f["area_mm2"], f["width_mm"],
                        f["height_mm"], f["zone"], f["x_mm"], f["y_mm"]))
    if sop_text:
        lines.append("SOP:\n" + sop_text)
    lines.append("Task: give 2-3 short operator actions consistent with the verdict.")
    return "\n".join(lines)


def inspect(heatmap, threshold, px_per_mm, rules=DEFAULT_RULES, **payload_kw):
    f = findings_from_heatmap(heatmap, threshold, px_per_mm, rules)
    v = verdict_from_findings(f, rules)
    return {"verdict": v, "findings": f,
            "llm_payload": build_llm_payload(f, v, **payload_kw)}
