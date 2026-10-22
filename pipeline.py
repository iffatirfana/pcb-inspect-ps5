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

from rules.engine import RuleEngine

CFG = yaml.safe_load(open("config.yaml", encoding="utf-8"))
RULES = RuleEngine(str(Path(CFG["paths"]["kb_dir"]) / "acceptance.yaml"))


# 0. Capture  [STUB - Person 1, real: Picamera2]
def capture():
    return np.zeros((480, 640, 3), dtype=np.uint8)


# 1. Preprocess  [STUB - real: preprocess/transforms.normalize_lighting]
def preprocess(img):
    return img


# 2. Align to golden board  [STUB - Person 1, real: ORB + homography]
def align(img):
    return img, np.eye(3)


# 3. Vision: anomaly + classifier  [STUB - Person 2, real: ONNX / Hailo]
# Contract: list of {type, conf, bbox[x1,y1,x2,y2], area_px}
def detect(img):
    return [
        {"type": "solder_bridge", "conf": 0.94, "bbox": [312, 210, 330, 224], "area_px": 252},
        {"type": "tombstone", "conf": 0.88, "bbox": [120, 80, 140, 96], "area_px": 210},
    ]


# 4. Board mapping  [STUB - Person 1, real: pick-and-place + netlist]
def map_to_board(defects, H):
    fake = [
        {"refdes": "U3", "pins": [4, 5], "nets": ["3V3", "GND"], "package": "QFN-32"},
        {"refdes": "R7", "pins": [1, 2], "nets": ["VOUT", "FB"], "package": "R_0603"},
    ]
    return [{**d, **fake[i % len(fake)]} for i, d in enumerate(defects)]


# 5. Rule engine  [REAL - rules/engine.py + kb/acceptance.yaml]
def apply_rules(defects):
    return [RULES.evaluate(d) for d in defects]


# 6. Llama advice  [STUB - Person 1, real: ExecuTorch .pte]
def explain(defects):
    verdict = RULES.board_verdict(defects)
    if not defects:
        return {"summary": "No defects found", "actions": [], "verdict": verdict}
    lines = [f"{d['type']} on {d['refdes']} ({d['severity']}, {d['verdict']})" for d in defects]
    actions = []
    for d in defects:
        if d.get("rule_note"):
            actions.append(f"{d['refdes']}: {d['rule_note']}")
        if d.get("sop_id"):
            actions.append(f"{d['refdes']}: follow {d['sop_id']}")
    return {"summary": "; ".join(lines), "actions": actions, "verdict": verdict}


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
