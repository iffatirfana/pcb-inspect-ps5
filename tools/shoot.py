"""Capture photos with no preview window. Press Enter in the terminal to take each photo.

  python tools/shoot.py --prefix pat_good_            # normal pattern-cloth shots
  python tools/shoot.py --prefix pat_miss_ --backend msmf --no-mjpg
Files are saved to data/photos/<prefix>NN.png, numbering continues from the last one. Type q + Enter to stop.
"""
import argparse, time
from pathlib import Path
import sys
import cv2
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fabric_calib as fcal

CAL = fcal.load().get("blue")      # same camera + table for every cloth


def save_both(p, f):
    """Save the raw photo and, if a calibration exists, a straightened copy in data/photos_corrected/."""
    cv2.imwrite(str(p), f)
    msg = "saved %s" % p
    if CAL:
        d = p.parent.parent / "photos_corrected"; d.mkdir(exist_ok=True)
        cv2.imwrite(str(d / p.name), fcal.apply(f, CAL)); msg += " + corrected copy"
    print(msg, "brightness", round(float(f.mean()), 1))

ap = argparse.ArgumentParser()
ap.add_argument("--prefix", required=True)
ap.add_argument("--index", type=int, default=0)
ap.add_argument("--backend", default="dshow", choices=["dshow", "msmf", "any"])
ap.add_argument("--no-mjpg", action="store_true")
ap.add_argument("--out", default="data/photos")
ap.add_argument("--live", action="store_true", help="show a live preview window: SPACE = save, q = quit")
a = ap.parse_args()

api = {"dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF, "any": cv2.CAP_ANY}[a.backend]
cap = cv2.VideoCapture(a.index, api)
if not a.no_mjpg:
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
if not cap.isOpened():
    raise SystemExit("Camera did not open. Try --index 1 or --backend msmf")
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
import re
_nums = [int(m.group(1)) for q in out.glob(a.prefix + "*.png") for m in [re.search(r"(\d+)\.png$", q.name)] if m]
n = max(_nums) if _nums else 0     # continue after the highest number, never overwrite
print("Camera ready, size", int(cap.get(3)), "x", int(cap.get(4)), "- next file number", n + 1)

if a.live:
    print("Live preview: click the window, SPACE = take photo, q = quit. Next file number", n + 1)
    for _ in range(15):
        cap.read()
    while True:
        ok, f = cap.read()
        if not ok or f is None:
            continue
        v = cv2.resize(f, (960, 540))
        cv2.putText(v, "%s%02d  SPACE=save  q=quit" % (a.prefix, n + 1), (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imshow("camera", v)
        k = cv2.waitKey(30) & 0xFF
        if k == ord("q"):
            break
        if k == 32:
            n += 1
            p = out / ("%s%02d.png" % (a.prefix, n))
            save_both(p, f)
    cap.release(); cv2.destroyAllWindows(); raise SystemExit
while True:
    s = input("Enter = take photo, q = quit > ").strip().lower()
    if s == "q":
        break
    for _ in range(15):          # let exposure settle, keep the last frame
        ok, f = cap.read()
        time.sleep(0.03)
    if not ok or f is None or f.mean() < 3:
        print("Black or empty frame - try --backend msmf --no-mjpg"); continue
    n += 1
    p = out / ("%s%02d.png" % (a.prefix, n))
    save_both(p, f)
cap.release()
