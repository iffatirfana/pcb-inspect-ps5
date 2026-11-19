"""Straighten every photo in data/photos with kb/calibration.json -> data/photos_corrected/ (raw files untouched).
  python tools/correct_photos.py
"""
import sys
from pathlib import Path
import cv2
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fabric_calib as fcal

cal = fcal.load().get("blue")
if not cal:
    raise SystemExit("No calibration found in kb/calibration.json")
src, dst = Path("data/photos"), Path("data/photos_corrected")
dst.mkdir(exist_ok=True)
n = 0
for p in sorted(src.glob("*.png")):
    img = cv2.imread(str(p))
    if img is None:
        continue
    cv2.imwrite(str(dst / p.name), fcal.apply(img, cal)); n += 1
print("corrected", n, "photos ->", dst)
