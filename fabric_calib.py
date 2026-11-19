"""Camera calibration for a tilted or off-centre camera: one scale for the whole frame.

A sheet of known size (A4 = 210 x 297 mm) is photographed flat on the cloth surface. Its four
corners give a homography that turns the tilted view into a straight top-down view at a fixed
scale, so px_per_mm is the same at the top and the bottom of the frame. The result is stored in
kb/calibration.json (written by tools/calibrate.py) and applied to every photo before detection.
"""
import json
from pathlib import Path

import cv2
import numpy as np

CAL_PATH = Path("kb/calibration.json")
MAX_SIDE = 2600           # largest output side in pixels


def build_calibration(corners_px, size_mm, ppm, frame_wh):
    """corners_px: 4 clicked corners of the sheet as seen in the photo, in the order top-left,
    top-right, bottom-right, bottom-left. size_mm = (width, height) of the sheet, width being the
    top-left -> top-right edge. ppm = wanted pixels per mm in the corrected image.
    Returns {"homography", "warp_size", "px_per_mm", "calibrated"}."""
    src = np.array(corners_px, np.float32).reshape(4, 2)
    w_mm, h_mm = size_mm
    dst = np.array([[0, 0], [w_mm, 0], [w_mm, h_mm], [0, h_mm]], np.float32) * float(ppm)
    h0 = cv2.getPerspectiveTransform(src, dst)
    fw, fh = frame_wh
    frame = np.array([[[0, 0], [fw - 1, 0], [fw - 1, fh - 1], [0, fh - 1]]], np.float32)
    c = cv2.perspectiveTransform(frame, h0)[0]                   # frame corners after correction
    x0, x1 = max(c[0][0], c[3][0]), min(c[1][0], c[2][0])        # largest axis-aligned rectangle
    y0, y1 = max(c[0][1], c[1][1]), min(c[2][1], c[3][1])        # that holds only real picture
    if x1 - x0 < 50 or y1 - y0 < 50:
        raise ValueError("The corrected picture is too small; check the corner order and the sheet size.")
    t = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], np.float64)
    h = t @ h0.astype(np.float64)
    size = (int(round(x1 - x0)), int(round(y1 - y0)))
    if max(size) > MAX_SIDE:
        raise ValueError("Corrected picture would be %dx%d; use a smaller --ppm." % size)
    return {"homography": h.tolist(), "warp_size": list(size), "px_per_mm": float(ppm), "calibrated": True}


def apply(img, cal):
    """Straighten a photo with a calibration dict (or its homography / warp_size fields). No-op without one."""
    if not cal or not cal.get("homography"):
        return img
    return cv2.warpPerspective(img, np.array(cal["homography"], np.float64), tuple(cal["warp_size"]),
                               flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def load(path=CAL_PATH):
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def save(fabric, cal, path=CAL_PATH):
    data = load(path)
    data[fabric] = cal
    Path(path).write_text(json.dumps(data, indent=1), encoding="utf-8")
