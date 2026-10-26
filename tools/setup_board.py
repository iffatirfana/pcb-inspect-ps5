"""Finish setting up an assembled board: compute its golden-diff threshold.

Put good-board photos (taken in the fixture) in the board folder first:
  kb/boards/<board>/golden.png   1 photo (alignment + main reference)
  kb/boards/<board>/refs/        ~4 photos (extra references)
  kb/boards/<board>/calib/       5-10 photos (only for the threshold, never references)
Then:  python -m tools.setup_board <board>
Writes anomaly_threshold into profile.yaml.
"""
import re
import sys
from pathlib import Path

import cv2

from align.register import AlignError
from mapping.profiles import list_profiles
from preprocess.transforms import load_config
from vision.golden_diff import GoldenDiff

IMG = (".png", ".jpg", ".jpeg", ".bmp")


def main(board_id, margin=1.1):
    cfg = load_config()
    profile = list_profiles(cfg["paths"]["boards_dir"])[board_id]
    calib = sorted(p for p in (profile.dir / "calib").glob("*") if p.suffix.lower() in IMG)
    if not calib:
        sys.exit(f"put 5-10 good-board photos in {profile.dir / 'calib'}")
    a = cfg["anomaly"]
    gd = GoldenDiff(tile=a["tile_size"], overlap=a["tile_overlap"], onnx=a.get("backbone_onnx"),
                    work_width=a.get("work_width"))
    golden = cv2.imread(str(profile.golden_path))
    refs = [golden] + [profile.aligner.align(cv2.imread(str(p)))[0] for p in profile.references]
    gd.set_references(refs)
    imgs, valids = [], []
    for p in calib:
        raw = cv2.imread(str(p))
        try:
            al, H, _ = profile.aligner.align(raw)
        except AlignError as e:
            print(f"skip {p.name}: {e}")
            continue
        imgs.append(al)
        valids.append(GoldenDiff.valid_mask(H, raw.shape, golden.shape))
    thr = gd.calibrate(imgs, valids, margin)
    y = profile.dir / "profile.yaml"
    text = y.read_text(encoding="utf-8")
    line = f"anomaly_threshold: {thr:.5f}"
    text = (re.sub(r"^anomaly_threshold:.*$", line, text, flags=re.M)
            if re.search(r"^anomaly_threshold:", text, re.M) else text.rstrip() + "\n" + line + "\n")
    y.write_text(text, encoding="utf-8")
    print(f"{board_id}: {len(refs)} references, {len(imgs)} calibration photos -> {line}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python -m tools.setup_board <board_id>")
    main(sys.argv[1])
