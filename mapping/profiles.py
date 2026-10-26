"""Board profiles: one shared pipeline, per-board reference data.

Each board design has a folder kb/boards/<board_id>/ containing:
  profile.yaml   name, board_type, golden image, calibration, references
  pos.csv        component centres (from tools/kicad_to_csv.py)
  pads.csv       pad positions + nets
  golden.png     known-good reference photo from the rig
  refs/*.png     extra good photos (assembled boards: golden-diff references)
Adding a new board = new folder; no model retraining for alignment, mapping,
rules or Llama.

board_type decides the inspector:
  assembled -> golden-diff (anomaly vs golden + refs) + MobileNetV3 classifier
  bare      -> YOLO (bare-board surface defects)
  detector: demo  (optional) -> fixed example findings, for tests and laptop demos
"""
from pathlib import Path

import cv2
import yaml

from align.register import Aligner
from mapping.board_map import BoardMap, image_to_board


class ProfileError(RuntimeError):
    pass


BOARD_TYPES = ("assembled", "bare")


class BoardProfile:
    def __init__(self, folder):
        self.dir = Path(folder)
        self.id = self.dir.name
        self.meta = yaml.safe_load((self.dir / "profile.yaml").read_text(encoding="utf-8")) or {}
        self.name = self.meta.get("name", self.id)
        cal = self.meta.get("calibration") or {}
        self.mm_per_px = cal.get("mm_per_px")
        self.origin_px = tuple(cal.get("origin_px") or (0, 0))
        self.rotation_deg = float(cal.get("rotation_deg") or 0.0)
        self.flip_y = bool(cal.get("flip_y", False))
        self.board_type = self.meta.get("board_type", "assembled")
        self.detector = self.meta.get("detector")          # None = by board_type
        self.anomaly_threshold = self.meta.get("anomaly_threshold")
        self._aligner = self._board_map = None

    def _path(self, key):
        v = self.meta.get(key)
        return self.dir / v if v else None

    @property
    def golden_path(self):
        return self._path("golden")

    @property
    def sample_image(self):
        p = self._path("sample_image")
        return p if p and p.exists() else None

    @property
    def references(self):
        """Extra good-board photos (refs/*.png|jpg), golden not included."""
        pattern = self.meta.get("references", "refs/*")
        return sorted(p for p in self.dir.glob(pattern)
                      if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp"))

    @property
    def inspector(self):
        """Which inspector runs for this board: demo | golden_diff | yolo."""
        if self.detector:
            return self.detector
        return "yolo" if self.board_type == "bare" else "golden_diff"

    def problems(self):
        """What is still missing before this board can be inspected."""
        out = []
        for f in ("pos.csv", "pads.csv"):
            if not (self.dir / f).exists():
                out.append(f"missing {f}")
        if not self.golden_path or not self.golden_path.exists():
            out.append("missing golden image")
        if not self.mm_per_px:
            out.append("camera calibration (mm_per_px) not set")
        if self.board_type not in BOARD_TYPES:
            out.append(f"board_type must be one of {BOARD_TYPES}")
        if self.inspector == "golden_diff" and not self.anomaly_threshold:
            out.append("anomaly_threshold not set (run: python -m tools.setup_board "
                       f"{self.id})")
        return out

    @property
    def ready(self):
        return not self.problems()

    @property
    def aligner(self):
        if self._aligner is None:
            if not self.golden_path or not self.golden_path.exists():
                raise ProfileError(f"board '{self.id}': missing golden image")
            img = cv2.imread(str(self.golden_path))
            if img is None:
                raise ProfileError(f"cannot read {self.golden_path}")
            self._aligner = Aligner(img)
        return self._aligner

    @property
    def board_map(self):
        if self._board_map is None:
            missing = [p for p in self.problems() if "csv" in p or "mm_per_px" in p]
            if missing:
                raise ProfileError(f"board '{self.id}': " + "; ".join(missing))
            self._board_map = BoardMap(self.dir, image_to_board(
                self.mm_per_px, self.origin_px, self.rotation_deg, self.flip_y))
        return self._board_map


def list_profiles(boards_dir="kb/boards"):
    root = Path(boards_dir)
    return {p.id: p for p in (BoardProfile(d) for d in sorted(root.iterdir())
                              if (d / "profile.yaml").exists())}


if __name__ == "__main__":
    for pid, p in list_profiles().items():
        state = "READY" if p.ready else "not ready: " + "; ".join(p.problems())
        print(f"{pid:20} {p.board_type:9} {p.inspector:11} {p.name:32} {state}")
