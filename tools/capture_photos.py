"""Take calibration and test photos from a USB webcam at 1920x1080 and name them by class.

Run from the repo root:   python tools/capture_photos.py            (camera 0)
                          python tools/capture_photos.py --index 1  (another camera)
Keys:  r ruler   g good   h hole   s stain   t thread   o other   q quit
Photos are saved in data/photos/ as ruler_01.png, good_01.png, hole_01.png, ... (PNG, no compression loss).
The preview window is small; the saved file is always the full camera frame.
"""
import argparse
import sys
from pathlib import Path

import cv2

KEYS = {ord("r"): "ruler", ord("g"): "good", ord("h"): "hole", ord("s"): "stain",
        ord("t"): "thread", ord("o"): "other"}


def next_name(folder, cls):
    n = 1
    while (folder / f"{cls}_{n:02d}.png").exists():
        n += 1
    return folder / f"{cls}_{n:02d}.png"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--out", default="data/photos")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(a.index, cv2.CAP_DSHOW) if sys.platform.startswith("win") else cv2.VideoCapture(a.index)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    if not cap.isOpened():
        sys.exit("Camera not found. Try --index 1.")
    ok, frame = cap.read()
    if not ok:
        sys.exit("Camera gave no frame.")
    h, w = frame.shape[:2]
    print(f"Camera size: {w}x{h}" + ("" if (w, h) == (1920, 1080) else "   WARNING: not 1920x1080"))
    counts = {c: len(list(out.glob(f"{c}_*.png"))) for c in KEYS.values()}
    msg = ""
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        view = cv2.resize(frame, (960, 540))
        line = "r ruler  g good  h hole  s stain  t thread  o other  q quit   " + "  ".join(
            f"{c}:{n}" for c, n in counts.items())
        cv2.putText(view, line, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(view, msg, (10, 520), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("capture", view)
        k = cv2.waitKey(30) & 0xFF
        if k == ord("q"):
            break
        if k in KEYS:
            for _ in range(2):           # take the newest frame
                ok, frame = cap.read()
            path = next_name(out, KEYS[k])
            cv2.imwrite(str(path), frame)
            counts[KEYS[k]] += 1
            msg = f"saved {path.name} ({frame.shape[1]}x{frame.shape[0]})"
            print(msg)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
