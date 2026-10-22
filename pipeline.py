"""PS5 inspection pipeline - end-to-end skeleton.

Stages marked STUB return data in the agreed format; replace one at a time
with the real module and the rest keeps working.
Run:  python pipeline.py
"""
import json
import sqlite3
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

import cv2

from align.register import Aligner
from llm.advisor import Advisor, MockBackend
from mapping.board_map import BoardMap, scale_transform
from rules.engine import RuleEngine

CFG = yaml.safe_load(open("config.yaml", encoding="utf-8"))
RULES = RuleEngine(str(Path(CFG["paths"]["kb_dir"]) / "acceptance.yaml"))
SAMPLE = Path("kb/board_sample")
ALIGNER = Aligner(cv2.imread(str(SAMPLE / "golden.png")))  # real: CFG golden_image
BOARD = BoardMap("kb/board_sample", scale_transform(0.05))  # real board: kb/board_demo + calibrated transform
ADVISOR = Advisor(MockBackend(), kb_dir=CFG["paths"]["kb_dir"])  # Pi: swap in ExecuTorch backend


# 0. Capture  [STUB - Person 1, real: Picamera2 on the Pi]
def capture():
    return cv2.imread(str(SAMPLE / "test_board.png"))


# 1. Preprocess  [STUB - real: preprocess/transforms.normalize_lighting]
def preprocess(img):
    return img


# 2. Align to golden board  [REAL - align/register.py]
def align(img):
    aligned, H, stats = ALIGNER.align(img)
    print("Alignment:", stats)
    return aligned, H


# 3. Vision: anomaly + classifier  [STUB - Person 2, real: ONNX / Hailo]
# Contract: list of {type, conf, bbox[x1,y1,x2,y2], area_px}
def detect(img):
    return [
        {"type": "solder_bridge", "conf": 0.94, "bbox": [348, 292, 356, 308], "area_px": 128},
        {"type": "tombstone", "conf": 0.88, "bbox": [680, 190, 720, 210], "area_px": 800},
    ]


# 4. Board mapping  [REAL - mapping/board_map.py, sample board for now]
def map_to_board(defects, H):
    return BOARD.annotate(defects)


# 5. Rule engine  [REAL - rules/engine.py + kb/acceptance.yaml]
def apply_rules(defects):
    return [RULES.evaluate(d) for d in defects]


# 6. Llama advice  [REAL logic, MOCK model - swap backend on the Pi]
def explain(defects):
    verdict = RULES.board_verdict(defects)
    advice, source = ADVISOR.advise(defects, verdict)
    return {**advice.model_dump(), "source": source}


# 7. UI  [STUB - real: Flask page]
def show(defects, advice):
    print(json.dumps({"defects": defects, "advice": advice}, indent=2))


# 8. Log to SQLite  [REAL]
def log(defects, advice, timings):
    Path(CFG["paths"]["db"]).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(CFG["paths"]["db"])
    con.execute("""CREATE TABLE IF NOT EXISTS inspections(
        ts TEXT, verdict TEXT, defects TEXT, advice TEXT, timings TEXT)""")
    con.execute("INSERT INTO inspections VALUES (?,?,?,?,?)",
                (datetime.now().isoformat(timespec="seconds"), advice["verdict"],
                 json.dumps(defects), json.dumps(advice), json.dumps(timings)))
    con.commit()
    con.close()


def run_once():
    t, timings = time.perf_counter(), {}

    def lap(name):
        nonlocal t
        now = time.perf_counter()
        timings[name] = round((now - t) * 1000, 2)
        t = now

    img = capture();                     lap("capture")
    img = preprocess(img);               lap("preprocess")
    img, H = align(img);                 lap("align")
    defects = detect(img);               lap("vision")
    defects = map_to_board(defects, H);  lap("mapping")
    defects = apply_rules(defects);      lap("rules")
    advice = explain(defects);           lap("llm")
    show(defects, advice);               lap("ui")
    log(defects, advice, timings)
    print("Board verdict:", advice["verdict"])
    print("Stage timings (ms):", timings)


if __name__ == "__main__":
    run_once()
