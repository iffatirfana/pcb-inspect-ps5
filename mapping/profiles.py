"""Board profiles: one shared pipeline, per-board reference data.

Each board design has a folder kb/boards/<board_id>/ containing:
  profile.yaml   name, golden image, calibration, anomaly memory bank
  pos.csv        component centres (from tools/kicad_to_csv.py)
  pads.csv       pad positions + nets
  golden.png     known-good reference photo from the rig
Adding a new board = new folder; no model retraining for alignment, mapping,
rules or Llama.
"""
from pathlib import Path

import cv2
import yaml

from align.register import Aligner
from mapping.board_map import BoardMap, scale_transform


class ProfileError(RuntimeError):
    pass


class BoardProfile:
    def __init__(self, folder):
        self.dir = Path(folder)
        self.id = self.dir.name
        self.meta = yaml.safe_load((self.dir / "profile.yaml").read_text(encoding="utf-8")) or {}
        self.name = self.meta.get("name", self.id)
        cal = self.meta.get("calibration") or {}
        self.mm_per_px = cal.get("mm_per_px")
        self.origin_px = tuple(cal.get("origin_px") or (0, 0))
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
    def anomaly_bank(self):
        return self._path("anomaly_bank")

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
        return out

    @property
    def ready(self):
        return not self.problems()

    @property
    def aligner(self):
        if self._aligner is None:
            self._require_ready()
            img = cv2.imread(str(self.golden_path))
            if img is None:
                raise ProfileError(f"cannot read {self.golden_path}")
            self._aligner = Aligner(img)
        return self._aligner

    @property
    def board_map(self):
        if self._board_map is None:
            self._require_ready()
            self._board_map = BoardMap(self.dir, scale_transform(self.mm_per_px, self.origin_px))
        return self._board_map

    def _require_ready(self):
        if not self.ready:
            raise ProfileError(f"board '{self.id}' not ready: " + "; ".join(self.problems()))


def list_profiles(boards_dir="kb/boards"):
    root = Path(boards_dir)
    return {p.id: p for p in (BoardProfile(d) for d in sorted(root.iterdir())
                              if (d / "profile.yaml").exists())}


if __name__ == "__main__":
    for pid, p in list_profiles().items():
        state = "READY" if p.ready else "not ready: " + "; ".join(p.problems())
        print(f"{pid:20} {p.name:32} {state}")
