"""Calibrate the camera from one photo of a sheet of known size (A4 by default).

1. Lay an A4 sheet FLAT on the cloth surface, in the middle of the frame, and take one photo
   with the final camera position (python tools/capture_photos.py, press r).
2. python tools/calibrate.py data/photos/ruler_01.png
3. Click the 4 corners of the sheet in this order: top-left, top-right, bottom-right, bottom-left
   (as you see it in the picture). Press Enter to accept, r to start again, q to quit.
   The top-left -> top-right edge is the sheet's WIDTH (--width-mm, default 210; --height-mm 297).
The tool saves kb/calibration.json (scale plus perspective correction) and a preview picture of the
corrected view. In that preview a ruler must show equal spacing everywhere; if not, calibrate again.
Do not move the camera afterwards: the correction is valid only for this camera position.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fabric_calib as fcal  # noqa: E402


def click_corners(img):
    scale = min(1.0, 1280 / img.shape[1])
    view0 = cv2.resize(img, None, fx=scale, fy=scale)
    pts = []

    def on_mouse(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
            pts.append((x / scale, y / scale))

    cv2.namedWindow("corners")
    cv2.setMouseCallback("corners", on_mouse)
    while True:
        view = view0.copy()
        for i, (x, y) in enumerate(pts):
            cv2.circle(view, (int(x * scale), int(y * scale)), 5, (0, 0, 255), -1)
            cv2.putText(view, "TL TR BR BL".split()[i], (int(x * scale) + 8, int(y * scale) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(view, "click TL, TR, BR, BL | Enter accept | r redo | q quit", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("corners", view)
        k = cv2.waitKey(30) & 0xFF
        if k == ord("q"):
            sys.exit("cancelled")
        if k == ord("r"):
            pts.clear()
        if k in (13, 10) and len(pts) == 4:
            break
    cv2.destroyAllWindows()
    return pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--fabric", default="blue")
    ap.add_argument("--width-mm", type=float, default=210.0)
    ap.add_argument("--height-mm", type=float, default=297.0)
    ap.add_argument("--ppm", type=float, default=5.0, help="pixels per mm wanted in the corrected image")
    ap.add_argument("--points", help="x1,y1,x2,y2,x3,y3,x4,y4 instead of clicking")
    a = ap.parse_args()
    img = cv2.imread(a.image)
    if img is None:
        sys.exit("Cannot read " + a.image)
    h, w = img.shape[:2]
    if a.points:
        v = [float(x) for x in a.points.split(",")]
        pts = list(zip(v[0::2], v[1::2]))
    else:
        pts = click_corners(img)
    cal = fcal.build_calibration(pts, (a.width_mm, a.height_mm), a.ppm, (w, h))
    fcal.save(a.fabric, cal)
    out = fcal.apply(img, cal)
    prev = Path(a.image).with_name(Path(a.image).stem + "_corrected.png")
    cv2.imwrite(str(prev), out)
    print("Saved kb/calibration.json for fabric '%s'" % a.fabric)
    print("px_per_mm = %.2f, corrected size = %dx%d (was %dx%d)" % (cal["px_per_mm"], *cal["warp_size"], w, h))
    print("Preview:", prev, "- check that equal lengths look equal everywhere in it.")


if __name__ == "__main__":
    main()
