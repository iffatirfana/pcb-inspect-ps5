"""Compare detector models on labelled photos (run on the laptop or the Pi).

    python tools/compare_models.py --normal photos/normal --defect photos/defect \
        --models models/best_a2.onnx models/best_169.onnx [--conf 0.5]

--normal and --defect are folders of photos that are already straightened and, ideally, were not used for
training. For each model it reports how many normal photos get a false alarm, how many defect photos are
detected, and the median detection time. Use the same photos for every model and keep the numbers with the
photo counts: small sets show that a method works, not how accurate it is.
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fabric_yolo import YoloDetector  # noqa: E402


def load(folder):
    files = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    return [(p.name, cv2.imread(str(p))) for p in files]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--normal", required=True)
    p.add_argument("--defect", required=True)
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--conf", type=float, default=0.5)
    a = p.parse_args()
    normal, defect = load(a.normal), load(a.defect)
    print("Normal photos: %d, defect photos: %d, confidence %.2f\n" % (len(normal), len(defect), a.conf))
    print("| Model | Input size | False alarms on normals | Defect photos detected | Median detect time (ms) |")
    print("|---|---|---|---|---|")
    for m in a.models:
        det = YoloDetector(m, conf=a.conf)
        det.detect(normal[0][1] if normal else defect[0][1])          # warm-up
        times, false_alarm, found, missed = [], 0, 0, []
        for name, img in normal:
            t = time.perf_counter(); d = det.detect(img); times.append((time.perf_counter() - t) * 1000)
            false_alarm += bool(d)
        for name, img in defect:
            t = time.perf_counter(); d = det.detect(img); times.append((time.perf_counter() - t) * 1000)
            found += bool(d)
            if not d:
                missed.append(name)
        print("| %s | %dx%d | %d of %d | %d of %d | %.0f |" % (Path(m).name, det.in_w, det.in_h, false_alarm, len(normal),
                                                           found, len(defect), float(np.median(times))))
        if missed:
            print("|  | missed: %s | | | |" % ", ".join(missed[:8]))


if __name__ == "__main__":
    main()
