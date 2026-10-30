"""PS5 inspection pipeline - one shared pipeline, inspector chosen per board.

    capture -> quality gate -> preprocess -> align (+ alignment check)
      -> inspector by board_type  (assembled: golden-diff + classifier | bare: YOLO)
      -> map to board -> rules -> VERDICT (shown + logged immediately)
      -> Llama advice (afterwards; never changes the verdict)

Verdicts: PASS | REVIEW | REWORK | REJECT, or INVALID_CAPTURE from the quality gate.
Run:  python pipeline.py [board_id]
"""
import hashlib
import json
import sqlite3
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import yaml

from align.register import AlignError
from capture.camera import Camera
from llm.advisor import Advisor, MockBackend
from mapping.profiles import ProfileError, list_profiles
from rules.engine import RuleEngine
from vision.golden_diff import GoldenDiff
from vision.inspectors import InspectorError, get_inspector
from vision.quality import check_alignment, check_capture

CFG = yaml.safe_load(open("config.yaml", encoding="utf-8"))
RULES = RuleEngine(str(Path(CFG["paths"]["kb_dir"]) / "acceptance.yaml"))
PROFILES = list_profiles(CFG["paths"]["boards_dir"])
_CAM = CFG.get("camera", {})
CAMERA = Camera(width=_CAM.get("width", 4056), height=_CAM.get("height", 3040),
                source=_CAM.get("source", "auto"), usb_index=_CAM.get("usb_index", 0),
                usb_size=tuple(_CAM.get("usb_size", (640, 480))),
                usb_exposure=_CAM.get("usb_exposure"),
                usb_lock_wb=_CAM.get("usb_lock_wb", False))   # HQ camera / USB webcam; else file
INVALID = "INVALID_CAPTURE"


def _make_backend():
    mode, llm = CFG["llm"].get("backend", "auto"), CFG["llm"]
    if mode in ("auto", "executorch"):
        try:
            from llm.executorch_backend import ExecuTorchBackend
            return ExecuTorchBackend(pte=llm["pte"], tokenizer=llm["tokenizer"],
                                     max_new_tokens=llm.get("max_new_tokens", 150),
                                     temperature=llm.get("temperature", 0.0)), "executorch"
        except FileNotFoundError:
            if mode == "executorch":
                raise
    return MockBackend(), "mock"


BACKEND, BACKEND_NAME = _make_backend()
ADVISOR = Advisor(BACKEND, kb_dir=CFG["paths"]["kb_dir"])


def get_profile(board_id=None):
    board_id = board_id or CFG["board"]
    if board_id not in PROFILES:
        raise ProfileError(f"unknown board '{board_id}'; available: {', '.join(PROFILES)}")
    profile = PROFILES[board_id]
    if not profile.ready:
        raise ProfileError(f"board '{board_id}' not ready: " + "; ".join(profile.problems()))
    return profile


# ---------------------------------------------------------------- stages
def capture(profile):
    """Pi: HQ camera or USB webcam (camera.source). Elsewhere: the profile's sample image."""
    if CAMERA.cam is not None:
        return CAMERA.capture()
    if profile.sample_image:
        return cv2.imread(str(profile.sample_image))
    raise RuntimeError(f"no camera and no sample_image for board '{profile.id}'")


def preprocess(img):
    """STUB - real: preprocess.transforms.normalize_lighting (apply the same to golden/refs)."""
    return img


def align(img, profile):
    aligned, H, stats = profile.aligner.align(img)
    valid = GoldenDiff.valid_mask(H, img.shape, aligned.shape)
    return aligned, valid, stats


def map_to_board(findings, profile):
    return profile.board_map.annotate(findings)


def apply_rules(findings):
    return [RULES.evaluate(d) for d in findings]


COLORS = {"CRITICAL": (0, 0, 255), "MAJOR": (0, 140, 255), "MINOR": (0, 200, 0)}


def draw(img, defects, path="results/last_inspection.png"):
    out = img.copy()
    for d in defects:
        x1, y1, x2, y2 = d["bbox"]
        c = (200, 80, 200) if d.get("verdict") == "REVIEW" else COLORS.get(d.get("severity"), (255, 255, 0))
        cv2.rectangle(out, (x1 - 6, y1 - 6), (x2 + 6, y2 + 6), c, 2)
        cv2.putText(out, f"{d.get('refdes', '')} {d['type']}", (x1 - 6, max(12, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1, cv2.LINE_AA)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(path, out)
    return path


# ---------------------------------------------------------------- logging
def _db():
    Path(CFG["paths"]["db"]).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(CFG["paths"]["db"])
    con.execute("""CREATE TABLE IF NOT EXISTS inspections(
        ts TEXT, verdict TEXT, defects TEXT, advice TEXT, timings TEXT)""")
    cols = {r[1] for r in con.execute("PRAGMA table_info(inspections)")}
    for col in ("board", "meta"):                     # databases from older versions
        if col not in cols:
            con.execute(f"ALTER TABLE inspections ADD COLUMN {col} TEXT")
    con.execute("""CREATE TABLE IF NOT EXISTS verifications(
        inspection_id INTEGER, defect_index INTEGER, label TEXT, note TEXT, ts TEXT)""")
    return con


def _file_id(path):
    p = Path(path)
    if not p.exists():
        return None
    if p.stat().st_size < 5_000_000:
        return hashlib.sha1(p.read_bytes()).hexdigest()[:10]
    return f"{p.stat().st_size}B@{int(p.stat().st_mtime)}"     # big model files


def versions(profile):
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                                text=True, timeout=2).stdout.strip() or None
    except Exception:
        commit = None
    files = {"config": "config.yaml", "profile": profile.dir / "profile.yaml",
             "rules": Path(CFG["paths"]["kb_dir"]) / "acceptance.yaml",
             "backbone": CFG["anomaly"].get("backbone_onnx"), "classifier": CFG["classifier"].get("model"),
             "yolo": CFG["yolo"].get("model"), "llm": CFG["llm"]["pte"]}
    return {"git": commit, "inspector": profile.inspector, "board_type": profile.board_type,
            "llm_backend": BACKEND_NAME, **{k: _file_id(v) for k, v in files.items() if v}}


def log(rec):
    con = _db()
    cur = con.execute("INSERT INTO inspections (ts, verdict, defects, advice, timings, board, meta) "
                      "VALUES (?,?,?,?,?,?,?)",
                      (rec["ts"], rec["verdict"], json.dumps(rec["defects"]),
                       json.dumps(rec.get("advice")), json.dumps(rec["timings"]), rec["board"],
                       json.dumps({"versions": rec["versions"], "quality": rec["quality"],
                                   "invalid_reasons": rec["invalid_reasons"]})))
    con.commit()
    rid = cur.lastrowid
    con.close()
    return rid


def update_advice(rec):
    con = _db()
    con.execute("UPDATE inspections SET advice=?, timings=? WHERE rowid=?",
                (json.dumps(rec["advice"]), json.dumps(rec["timings"]), rec["id"]))
    con.commit()
    con.close()


def verify(inspection_id, defect_index, label, note=""):
    """Operator feedback: label = correct | wrong | missed."""
    con = _db()
    con.execute("INSERT INTO verifications VALUES (?,?,?,?,?)",
                (inspection_id, defect_index, label, note,
                 datetime.now().isoformat(timespec="seconds")))
    con.commit()
    con.close()


# ---------------------------------------------------------------- main entry points
def inspect(board_id=None, image=None):
    """Everything up to the verdict (fast). Returns the inspection record, already logged."""
    profile = get_profile(board_id)
    t, timings = time.perf_counter(), {}

    def lap(name):
        nonlocal t
        now = time.perf_counter()
        timings[name] = round((now - t) * 1000, 2)
        t = now

    rec = {"board": profile.id, "board_name": profile.name, "board_type": profile.board_type,
           "inspector": profile.inspector, "ts": datetime.now().isoformat(timespec="seconds"),
           "defects": [], "invalid_reasons": [], "quality": {}, "advice": None, "image": None}
    img = image if image is not None else capture(profile);  lap("capture")
    ok, reasons, rec["quality"] = check_capture(img, CFG.get("quality"));  lap("quality")
    if ok:
        img = preprocess(img);                                  lap("preprocess")
        try:
            img, valid, stats = align(img, profile)
            rec["quality"]["align"] = stats
            ok, reasons = check_alignment(stats, CFG.get("quality"))
        except AlignError as e:
            ok, reasons = False, [f"board not recognised: {e}"]
        lap("align")
    if ok:
        findings = get_inspector(profile, CFG).detect(img, valid);  lap("vision")
        defects = map_to_board(findings, profile);                lap("mapping")
        rec["defects"] = apply_rules(defects);                    lap("rules")
        rec["verdict"] = RULES.board_verdict(rec["defects"])
    else:
        rec["verdict"], rec["invalid_reasons"] = INVALID, reasons
    if img is not None:
        rec["image"] = draw(img, rec["defects"]);              lap("draw")
    timings["to_verdict"] = round(sum(timings.values()), 2)
    rec["timings"], rec["versions"] = timings, versions(profile)
    rec["id"] = log(rec)
    return rec


def advise(rec):
    """Llama explanation for an inspection record (slow; run after the verdict is shown)."""
    t = time.perf_counter()
    if rec["verdict"] == INVALID:
        advice = {"summary": "Photo not usable - no verdict given.",
                  "risk": "Inspection not performed.",
                  "actions": rec["invalid_reasons"] + ["Re-seat the board and inspect again."],
                  "verdict": INVALID, "source": "template"}
    else:
        adv, source = ADVISOR.advise(rec["defects"], rec["verdict"])
        advice = {**adv.model_dump(), "source": source}
    rec["advice"] = advice
    rec["timings"]["llm"] = round((time.perf_counter() - t) * 1000, 2)
    rec["timings"]["total"] = round(rec["timings"]["to_verdict"] + rec["timings"]["llm"], 2)
    update_advice(rec)
    return advice


def run_once(board_id=None, verbose=True, image=None):
    rec = inspect(board_id, image)
    advise(rec)
    if verbose:
        print(json.dumps({"defects": rec["defects"], "advice": rec["advice"]}, indent=2))
        print(f"Board: {rec['board']} ({rec['board_type']}, {rec['inspector']}) | "
              f"verdict: {rec['verdict']}")
        if rec["invalid_reasons"]:
            print("Invalid capture:", "; ".join(rec["invalid_reasons"]))
        print("Stage timings (ms):", rec["timings"])
    return rec


if __name__ == "__main__":
    run_once(sys.argv[1] if len(sys.argv) > 1 else None)
