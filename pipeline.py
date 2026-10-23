"""PS5 inspection pipeline - one shared pipeline, per-board profiles.

Stages marked STUB return data in the agreed format; replace one at a time
with the real module and the rest keeps working.
Run:  python pipeline.py [board_id]        (board_id = folder in kb/boards/)
"""
import json
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import yaml

from capture.camera import Camera
from llm.advisor import Advisor, MockBackend
from mapping.profiles import ProfileError, list_profiles
from rules.engine import RuleEngine

CFG = yaml.safe_load(open("config.yaml", encoding="utf-8"))
RULES = RuleEngine(str(Path(CFG["paths"]["kb_dir"]) / "acceptance.yaml"))
PROFILES = list_profiles(CFG["paths"]["boards_dir"])
CAMERA = Camera()                                     # Picamera2 on the Pi; None elsewhere
ADVISOR = Advisor(MockBackend(), kb_dir=CFG["paths"]["kb_dir"])  # Pi: swap in ExecuTorch backend


def get_profile(board_id=None):
    board_id = board_id or CFG["board"]
    if board_id not in PROFILES:
        raise ProfileError(f"unknown board '{board_id}'; available: {', '.join(PROFILES)}")
    profile = PROFILES[board_id]
    if not profile.ready:
        raise ProfileError(f"board '{board_id}' not ready: " + "; ".join(profile.problems()))
    return profile


# 0. Capture  [REAL - Picamera2 on the Pi; profile's sample image elsewhere]
def capture(profile):
    if CAMERA.cam is not None:
        return CAMERA.capture()
    if profile.sample_image:
        return cv2.imread(str(profile.sample_image))
    raise RuntimeError(f"no camera and no sample_image for board '{profile.id}'")


# 1. Preprocess  [STUB - real: preprocess/transforms.normalize_lighting]
def preprocess(img):
    return img


# 2. Align to the board's golden image  [REAL - align/register.py]
def align(img, profile):
    aligned, H, stats = profile.aligner.align(img)
    return aligned, H


# 3. Vision: anomaly + classifier  [STUB - Person 2, real: ONNX / Hailo]
# Contract: list of {type, conf, bbox[x1,y1,x2,y2], area_px} in aligned-image pixels.
# The real version tiles the image (config anomaly.tile_size / tile_overlap,
# preprocess.transforms.tile_image / stitch_maps) and uses profile.anomaly_bank.
def detect(img, profile):
    if profile.id != "sample_synthetic":
        return []                                     # stub only knows the synthetic board
    return [
        {"type": "solder_bridge", "conf": 0.94, "bbox": [348, 292, 356, 308], "area_px": 128},
        {"type": "tombstone", "conf": 0.88, "bbox": [680, 190, 720, 210], "area_px": 800},
    ]


# 4. Board mapping  [REAL - mapping/board_map.py with the profile's pos/pads]
def map_to_board(defects, profile):
    return profile.board_map.annotate(defects)


# 5. Rule engine  [REAL - rules/engine.py + kb/acceptance.yaml]
def apply_rules(defects):
    return [RULES.evaluate(d) for d in defects]


# 6. Llama advice  [REAL logic, MOCK model - swap backend on the Pi]
def explain(defects):
    verdict = RULES.board_verdict(defects)
    advice, source = ADVISOR.advise(defects, verdict)
    return {**advice.model_dump(), "source": source}


# 7. Output for UI  [REAL - draws boxes; ui/app.py shows it]
COLORS = {"CRITICAL": (0, 0, 255), "MAJOR": (0, 140, 255), "MINOR": (0, 200, 0)}


def draw(img, defects, path="results/last_inspection.png"):
    out = img.copy()
    for d in defects:
        x1, y1, x2, y2 = d["bbox"]
        c = COLORS.get(d.get("severity"), (255, 255, 0))
        cv2.rectangle(out, (x1 - 6, y1 - 6), (x2 + 6, y2 + 6), c, 2)
        cv2.putText(out, f"{d['refdes']} {d['type']}", (x1 - 6, max(12, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1, cv2.LINE_AA)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(path, out)
    return path


def show(defects, advice):
    print(json.dumps({"defects": defects, "advice": advice}, indent=2))


# 8. Log to SQLite  [REAL]
def log(board_id, defects, advice, timings):
    Path(CFG["paths"]["db"]).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(CFG["paths"]["db"])
    con.execute("""CREATE TABLE IF NOT EXISTS inspections(
        ts TEXT, verdict TEXT, defects TEXT, advice TEXT, timings TEXT, board TEXT)""")
    cols = [r[1] for r in con.execute("PRAGMA table_info(inspections)")]
    if "board" not in cols:                           # database from an older version
        con.execute("ALTER TABLE inspections ADD COLUMN board TEXT")
    con.execute("INSERT INTO inspections (ts, verdict, defects, advice, timings, board) "
                "VALUES (?,?,?,?,?,?)",
                (datetime.now().isoformat(timespec="seconds"), advice["verdict"],
                 json.dumps(defects), json.dumps(advice), json.dumps(timings), board_id))
    con.commit()
    con.close()


def run_once(board_id=None, verbose=True):
    profile = get_profile(board_id)
    t, timings = time.perf_counter(), {}

    def lap(name):
        nonlocal t
        now = time.perf_counter()
        timings[name] = round((now - t) * 1000, 2)
        t = now

    img = capture(profile);                    lap("capture")
    img = preprocess(img);                     lap("preprocess")
    img, H = align(img, profile);              lap("align")
    defects = detect(img, profile);            lap("vision")
    defects = map_to_board(defects, profile);  lap("mapping")
    defects = apply_rules(defects);            lap("rules")
    advice = explain(defects);                 lap("llm")
    image_path = draw(img, defects);           lap("draw")
    timings["total"] = round(sum(timings.values()), 2)
    log(profile.id, defects, advice, timings)
    if verbose:
        show(defects, advice)
        print("Board:", profile.id, "| verdict:", advice["verdict"])
        print("Stage timings (ms):", timings)
    return {"board": profile.id, "board_name": profile.name, "defects": defects,
            "advice": advice, "timings": timings, "image": image_path}


if __name__ == "__main__":
    run_once(sys.argv[1] if len(sys.argv) > 1 else None)
