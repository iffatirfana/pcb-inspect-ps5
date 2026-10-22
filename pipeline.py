"""PS5 inspection pipeline - end-to-end skeleton.

Every stage below is a STUB returning data in the agreed format.
Replace one stub at a time with the real module; the rest keeps working.
Run:  python pipeline.py
"""
import json
import sqlite3
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

CFG = yaml.safe_load(open("config.yaml", encoding="utf-8"))


# 0. Capture ---------------------------------------------- (Person 1, real: Picamera2)
def capture():
    return np.zeros((480, 640, 3), dtype=np.uint8)  # fake black image


# 1. Preprocess ------------------------------------------- (real: preprocess/transforms.py)
def preprocess(img):
    return img


# 2. Align to golden board -------------------------------- (Person 1, real: ORB + homography)
def align(img):
    H = np.eye(3)  # identity = "already aligned"
    return img, H


# 3. Vision: anomaly + classifier -------------------------- (Person 2, real: ONNX/Hailo)
def detect(img):
    # contract: list of {type, conf, bbox[x1,y1,x2,y2], area_px}
    return [{"type": "solder_bridge", "conf": 0.94,
             "bbox": [312, 210, 330, 224], "area_px": 252}]


# 4. Board mapping ----------------------------------------- (Person 1, real: pick-and-place + netlist)
def map_to_board(defects, H):
    for d in defects:
        d.update({"refdes": "U3", "pins": [4, 5],
                  "nets": ["3V3", "GND"], "package": "QFN-32"})
    return defects


# 5. Rule engine -------------------------------------------- (Person 1, real: kb/acceptance.yaml)
def apply_rules(defects):
    for d in defects:
        power_short = {"3V3", "GND"} <= set(d.get("nets", []))
        d.update({"severity": "CRITICAL" if power_short else "MAJOR",
                  "verdict": "REWORK",
                  "sop_id": "SOP-BRIDGE-01",
                  "power_short": power_short})
    return defects


# 6. Llama advice ------------------------------------------- (Person 1, real: ExecuTorch .pte)
def explain(defects):
    if not defects:
        return {"summary": "No defects found", "actions": [], "verdict": "PASS"}
    d = defects[0]
    return {"summary": f"{d['type']} on {d['refdes']} pins {d['pins']}",
            "risk": "Power-to-ground short" if d["power_short"] else "Possible functional fault",
            "actions": ["Do not power the board", "Follow " + d["sop_id"]],
            "verdict": d["verdict"]}


# 7. UI ----------------------------------------------------- (real: Flask page)
def show(defects, advice):
    print(json.dumps({"defects": defects, "advice": advice}, indent=2))


# 8. Log to SQLite (real already) ---------------------------
def log(defects, advice, timings):
    Path(CFG["paths"]["db"]).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(CFG["paths"]["db"])
    con.execute("""CREATE TABLE IF NOT EXISTS inspections(
        ts TEXT, verdict TEXT, defects TEXT, advice TEXT, timings TEXT)""")
    con.execute("INSERT INTO inspections VALUES (?,?,?,?,?)",
                (datetime.now().isoformat(timespec="seconds"),
                 advice.get("verdict", "PASS"),
                 json.dumps(defects), json.dumps(advice), json.dumps(timings)))
    con.commit()
    con.close()


def run_once():
    t, timings = time.perf_counter(), {}

    def lap(name):
        nonlocal t
        now = time.perf_counter()
        timings[name] = round((now - t) * 1000, 2)  # milliseconds
        t = now

    img = capture();                       lap("capture")
    img = preprocess(img);                 lap("preprocess")
    img, H = align(img);                   lap("align")
    defects = detect(img);                 lap("vision")
    defects = map_to_board(defects, H);    lap("mapping")
    defects = apply_rules(defects);        lap("rules")
    advice = explain(defects);             lap("llm")
    show(defects, advice);                 lap("ui")
    log(defects, advice, timings)
    print("Stage timings (ms):", timings)


if __name__ == "__main__":
    run_once()