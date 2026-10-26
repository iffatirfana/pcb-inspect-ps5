"""Board mapping: defect pixel bbox -> component, pins, nets (stage 4).

Inputs (per board, in kb/<board>/):
  pos.csv   - KiCad/Altium pick-and-place: Ref,Val,Package,PosX,PosY,Rot,Side (mm)
  pads.csv  - pad positions + nets: Ref,Pin,X,Y,Net (mm)   [optional but needed
              for pin/net-level reasoning]
Image -> board transform:
  img_to_mm - 3x3 matrix mapping golden-image pixels to board millimetres
              (from calibration: board size + fiducials). Test images are first
              aligned to the golden image (stage 2), so this is fixed per board.
"""
import csv
from collections import Counter
from pathlib import Path

import numpy as np


def _read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return [{k.strip(): (v or "").strip() for k, v in row.items()} for row in csv.DictReader(f)]


def scale_transform(mm_per_px, origin_px=(0.0, 0.0), flip_y=False, img_h=0):
    """Simple pixel->mm matrix for a fronto-parallel camera (no rotation)."""
    sy = -mm_per_px if flip_y else mm_per_px
    ty = img_h * mm_per_px if flip_y else 0.0
    return np.array([[mm_per_px, 0, -origin_px[0] * mm_per_px],
                     [0, sy, -origin_px[1] * sy + ty],
                     [0, 0, 1]], dtype=float)


def image_to_board(mm_per_px, origin_px=(0.0, 0.0), rotation_deg=0.0, flip_y=False):
    """Pixel -> board-mm matrix for a fronto-parallel camera.

    origin_px:    pixel (in the golden image) where board (0, 0) mm lies
    rotation_deg: board rotation in the image (CAD +x axis measured from image +x)
    flip_y:       True if CAD y grows upwards while image y grows downwards
    """
    t = np.radians(rotation_deg)
    R = np.array([[np.cos(t), np.sin(t), 0], [-np.sin(t), np.cos(t), 0], [0, 0, 1]])
    S = np.diag([mm_per_px, -mm_per_px if flip_y else mm_per_px, 1.0])
    T = np.array([[1, 0, -origin_px[0]], [0, 1, -origin_px[1]], [0, 0, 1]], dtype=float)
    return S @ R @ T


class BoardMap:
    def __init__(self, board_dir, img_to_mm, pad_margin_mm=0.1, comp_radius_mm=3.0):
        board_dir = Path(board_dir)
        self.parts = {r["Ref"]: {"value": r.get("Val", ""), "package": r.get("Package", ""),
                                 "x": float(r["PosX"]), "y": float(r["PosY"])}
                      for r in _read_csv(board_dir / "pos.csv")}
        pads_file = board_dir / "pads.csv"
        self.pads = [{"ref": r["Ref"], "pin": r["Pin"], "x": float(r["X"]), "y": float(r["Y"]),
                      "net": r.get("Net", "")}
                     for r in _read_csv(pads_file)] if pads_file.exists() else []
        self.T = np.asarray(img_to_mm, dtype=float)
        self.pad_margin = pad_margin_mm
        self.comp_radius = comp_radius_mm

    def px_to_mm(self, x, y):
        v = self.T @ np.array([x, y, 1.0])
        return v[0] / v[2], v[1] / v[2]

    def bbox_to_mm(self, bbox):
        x1, y1, x2, y2 = bbox
        pts = [self.px_to_mm(x, y) for x, y in ((x1, y1), (x2, y1), (x1, y2), (x2, y2))]
        xs, ys = zip(*pts)
        return min(xs), min(ys), max(xs), max(ys)

    def locate(self, bbox):
        """Return {refdes, pins, nets, package, value, x_mm, y_mm, match}."""
        bx1, by1, bx2, by2 = self.bbox_to_mm(bbox)
        cx, cy = (bx1 + bx2) / 2, (by1 + by2) / 2
        m = self.pad_margin
        hit = [p for p in self.pads
               if bx1 - m <= p["x"] <= bx2 + m and by1 - m <= p["y"] <= by2 + m]
        if hit:
            ref = Counter(p["ref"] for p in hit).most_common(1)[0][0]
            mine = sorted((p for p in hit if p["ref"] == ref), key=lambda p: p["pin"])
            pins = [int(p["pin"]) if p["pin"].isdigit() else p["pin"] for p in mine]
            nets = [p["net"] for p in mine]
            match = "pads"
        else:
            best = min(self.parts.items(),
                       key=lambda kv: (kv[1]["x"] - cx) ** 2 + (kv[1]["y"] - cy) ** 2,
                       default=(None, None))
            ref, part = best
            if part is None or ((part["x"] - cx) ** 2 + (part["y"] - cy) ** 2) ** 0.5 > self.comp_radius:
                return {"refdes": None, "pins": [], "nets": [], "package": "", "value": "",
                        "x_mm": round(cx, 2), "y_mm": round(cy, 2), "match": "none"}
            pins, nets, match = [], [], "component"
        part = self.parts.get(ref, {})
        return {"refdes": ref, "pins": pins, "nets": nets, "package": part.get("package", ""),
                "value": part.get("value", ""), "x_mm": round(cx, 2), "y_mm": round(cy, 2),
                "match": match}

    def annotate(self, defects):
        """Add board info to each defect dict from the vision stage."""
        out = []
        for d in defects:
            info = self.locate(d["bbox"])
            if info["refdes"] is None:
                info["refdes"] = f"@({info['x_mm']},{info['y_mm']})mm"
            out.append({**d, **info})
        return out


def self_test():
    # sample board: 0.05 mm per pixel (20 px/mm), image origin = board origin
    bm = BoardMap("kb/boards/sample_synthetic", scale_transform(0.05))
    tests = {
        "bridge U3 pins 4-5": [348, 292, 356, 308],   # 17.4-17.8 x 14.6-15.4 mm
        "tombstone on R7":    [680, 190, 720, 210],   # 34-36 x 9.5-10.5 mm
        "missing D1 (body)":  [150, 490, 170, 510],   # 7.5-8.5 x 24.5-25.5 mm, between pads
        "scratch empty area": [900, 900, 920, 905],   # 45 x 45 mm, nothing there
    }
    for name, bbox in tests.items():
        r = bm.locate(bbox)
        print(f"{name:20} -> {str(r['refdes']):5} pins={r['pins']} nets={r['nets']} "
              f"pkg={r['package']} match={r['match']}")


if __name__ == "__main__":
    self_test()
