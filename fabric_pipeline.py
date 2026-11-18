"""Fabric inspection pipeline (separate from the PCB pipeline.py; shares camera, quality gate, LLM backend, DB).

  capture -> quality gate -> crop -> anomaly heatmap -> findings (mm) -> rules -> VERDICT
          -> Llama advice (afterwards; never changes the verdict) -> SQLite log

Run:  python fabric_pipeline.py <fabric> [image.png]     (fabric: blue | white | purple)
"""
import json
import os
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml

import fabric_profile as fp
from fabric_quality import cloth_quality
from fabric_advisor import FabricAdvisor, FabricMockBackend, build_user_message
from vision.quality import check_capture

CFG = yaml.safe_load(open("config.yaml", encoding="utf-8"))
FAB = yaml.safe_load(open("kb/fabric.yaml", encoding="utf-8"))
INVALID = "INVALID_CAPTURE"
_scorers = {}
_detectors = {}
_anomaly = {}


def make_backend():
    llm = CFG["llm"]
    if llm.get("backend", "auto") in ("auto", "executorch"):
        try:
            from llm.executorch_backend import ExecuTorchBackend
            return ExecuTorchBackend(pte=llm["pte"], tokenizer=llm["tokenizer"],
                                     max_new_tokens=llm.get("max_new_tokens", 150),
                                     temperature=llm.get("temperature", 0.0)), "executorch"
        except Exception:
            if llm.get("backend") == "executorch":
                raise
    return FabricMockBackend(), "mock"


def get_scorer(fabric):
    if fabric not in _scorers:
        from fabric_scorer import FabricScorer
        c = FAB["fabrics"][fabric]
        _scorers[fabric] = FabricScorer(c["backbone_onnx"], c["bank"],
                                        FAB["scorer"]["work_size"], FAB["scorer"]["blur_ksize"])
    return _scorers[fabric]


def get_anomaly_scorer(fabric):
    """Second-opinion scorer (memory-bank heatmap), or None when disabled or its files are missing."""
    if fabric not in _anomaly:
        a = FAB["fabrics"][fabric].get("anomaly") or {}
        sc = None
        if a.get("enabled") and Path(a["backbone_onnx"]).exists() and Path(a["bank"]).exists():
            from fabric_scorer import FabricScorer
            sc = FabricScorer(a["backbone_onnx"], a["bank"], FAB["scorer"]["work_size"],
                              FAB["scorer"]["blur_ksize"])
        _anomaly[fabric] = sc
    return _anomaly[fabric]


def anomaly_second_opinion(img, fabric, px_per_mm, rules, scorer=None):
    """Run only when the detector found nothing. Returns None when unavailable, otherwise
    {"flagged": bool, "fraction": float, "max": float, "findings": [...]}.
    A flagged photo yields up to 3 findings of type 'Anomaly', always severity Low: the caller
    sets the verdict to REVIEW, so this check can never cause REWORK or REJECT."""
    a = FAB["fabrics"][fabric].get("anomaly") or {}
    scorer = scorer or get_anomaly_scorer(fabric)
    if scorer is None:
        return None
    hm = scorer.heatmap(img)
    fraction = float((hm > a["pixel_threshold"]).mean())
    out = {"flagged": fraction > a["image_fraction"], "fraction": round(fraction, 4),
           "max": round(float(hm.max()), 3), "findings": [], "hm": hm}
    if out["flagged"]:
        fnd = fp.findings_from_heatmap(hm, a["pixel_threshold"], px_per_mm, rules)[:3]
        if not fnd:                                   # flagged by many tiny regions: mark the peak
            y, x = np.unravel_index(int(hm.argmax()), hm.shape)
            fnd = [{"area_mm2": 0.0, "width_mm": 0.0, "height_mm": 0.0, "aspect": 1.0,
                    "x_mm": round(x / px_per_mm, 1), "y_mm": round(y / px_per_mm, 1),
                    "zone": fp.zone_3x3(x, y, hm.shape[1], hm.shape[0]), "peak_score": out["max"]}]
        frame_mm2 = hm.shape[0] * hm.shape[1] / (px_per_mm ** 2)
        for f in fnd:
            f.update(type="Anomaly", severity="Low", confidence=None)
            if f.get("area_mm2", 0.0) > 0.25 * frame_mm2:   # a region this big is lighting or fabric change, not a size
                f["wide"] = True
                f["area_mm2"] = 0.0
        out["findings"] = fnd
    return out


def get_detector(fabric):
    """YOLO detector for fabrics whose config says detector: yolo."""
    if fabric not in _detectors:
        from fabric_yolo import YoloDetector
        c = FAB["fabrics"][fabric]
        _detectors[fabric] = YoloDetector(c["yolo_model"], conf=c.get("yolo_conf", 0.5),
                                          imgsz=c.get("yolo_imgsz", 640))
    return _detectors[fabric]


def capture():
    from capture.camera import Camera
    c = CFG.get("camera", {})
    cam = Camera(width=c.get("width", 1920), height=c.get("height", 1080), source=c.get("source", "auto"),
                 usb_index=c.get("usb_index", 0), usb_size=tuple(c.get("usb_size", (1920, 1080))),
                 usb_exposure=c.get("usb_exposure"), usb_lock_wb=c.get("usb_lock_wb", False))
    if cam.cam is None:
        raise RuntimeError("no camera; pass an image path")
    return cam.capture()


def _db():
    Path(CFG["paths"]["db"]).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(CFG["paths"]["db"])
    con.execute("""CREATE TABLE IF NOT EXISTS fabric_inspections(
        ts TEXT, fabric TEXT, verdict TEXT, findings TEXT, advice TEXT, timings TEXT,
        operator_label TEXT)""")
    for col in ("reason", "note", "corrected", "quality"):      # added later: safe on old databases
        try:
            con.execute("ALTER TABLE fabric_inspections ADD COLUMN %s TEXT" % col)
        except sqlite3.OperationalError:
            pass
    con.execute("CREATE TABLE IF NOT EXISTS device_usage(session_start TEXT, seconds REAL)")
    return con


def log(rec):
    con = _db()
    cur = con.execute("INSERT INTO fabric_inspections(ts,fabric,verdict,findings,advice,timings,quality) "
                      "VALUES (?,?,?,?,?,?,?)",
                      (rec["ts"], rec["fabric"], rec["verdict"], json.dumps(rec["findings"]),
                       json.dumps(rec.get("advice")), json.dumps(rec["timings"]),
                       (rec.get("cloth_quality") or {}).get("score")))
    con.commit(); rid = cur.lastrowid; con.close()
    return rid


def operator_label(inspection_id, label, reason=None, note=None, corrected=None):
    """label: correct | wrong | missed. reason, note and corrected verdict are optional operator detail."""
    con = _db()
    con.execute("UPDATE fabric_inspections SET operator_label=?, reason=?, note=?, corrected=? WHERE rowid=?",
                (label, reason or None, (note or "")[:300] or None, corrected or None, inspection_id))
    con.commit(); con.close()


def usage_start():
    """Call once when the dashboard starts. Returns the session id used by usage_tick."""
    con = _db()
    cur = con.execute("INSERT INTO device_usage VALUES (?, 0)", (datetime.now().isoformat(timespec="seconds"),))
    con.commit(); sid = cur.lastrowid; con.close()
    return sid


def usage_tick(session_id, seconds):
    con = _db()
    con.execute("UPDATE device_usage SET seconds=? WHERE rowid=?", (float(seconds), session_id))
    con.commit(); con.close()


def usage_total_seconds():
    """Device use time summed over every dashboard session ever run on this device."""
    con = _db()
    total = con.execute("SELECT COALESCE(SUM(seconds), 0) FROM device_usage").fetchone()[0]
    con.close()
    return float(total)


def inspect(fabric=None, image=None, scorer=None, detector=None, anomaly_scorer=None):
    """Everything up to the verdict. Returns the record (already logged).
    Fabrics with detector: yolo use the YOLO model; others use the anomaly heatmap."""
    fabric = fabric or FAB["fabric"]
    c, rules = FAB["fabrics"][fabric], FAB["rules"]
    t, timings = time.perf_counter(), {}

    def lap(name):
        nonlocal t
        n = time.perf_counter(); timings[name] = round((n - t) * 1000, 2); t = n

    rec = {"fabric": fabric, "ts": datetime.now().isoformat(timespec="seconds"),
           "findings": [], "invalid_reasons": [], "advice": None}
    img = image if image is not None else capture();            lap("capture")
    ok, reasons, rec["quality"] = check_capture(img, {**(CFG.get("quality") or {}), **(FAB.get("quality") or {})});  lap("quality")
    if ok:
        img = fp.crop_image(img, c.get("crop"))
        hm_overlay = None
        if c.get("detector", "anomaly") == "yolo":
            dets = (detector or get_detector(fabric)).detect(img);   lap("detect")
            findings = fp.findings_from_detections(dets, img.shape, c["px_per_mm"], rules)
            if not findings and (c.get("anomaly") or {}).get("enabled"):       # second opinion, only when clean
                op = anomaly_second_opinion(img, fabric, c["px_per_mm"], rules, anomaly_scorer);   lap("anomaly")
                if op is not None:
                    a = c["anomaly"]
                    rec["anomaly"] = {k: op[k] for k in ("flagged", "fraction", "max")}
                    rec["anomaly"]["limit"] = a["image_fraction"]
                    findings = op["findings"]
                    hm_overlay = (op.get("hm"), a["pixel_threshold"])
        else:
            hm = (scorer or get_scorer(fabric)).heatmap(img);   lap("heatmap")
            findings = fp.findings_from_heatmap(hm, c["threshold"], c["px_per_mm"], rules)
        rec["uncalibrated"] = not c.get("calibrated", True)
        rec["findings"] = findings
        rec["verdict"] = ("REVIEW" if rec.get("anomaly", {}).get("flagged")          # anomaly-only: never above REVIEW
                          else fp.verdict_from_findings(findings, rules));  lap("rules")
        rec["image"] = _draw(img, findings, c["px_per_mm"])
        if hm_overlay and hm_overlay[0] is not None:       # picture of what the second opinion saw
            rec["heatmap_image"] = _draw_heatmap(img, *hm_overlay)
        lap("draw")
        rec["cloth_quality"] = cloth_quality(findings, rec["verdict"], rec.get("anomaly"))
    else:
        rec["verdict"], rec["invalid_reasons"] = INVALID, reasons
    timings["to_verdict"] = round(sum(timings.values()), 2)
    rec["timings"] = timings
    rec["id"] = log(rec)
    return rec


_COL = {"Low": (0, 200, 0), "Medium": (0, 140, 255), "High": (0, 0, 255)}


def _draw(img, findings, ppm, path="results/last_fabric.png"):
    out = img.copy()
    for f in findings:
        cx, cy = int(f["x_mm"] * ppm), int(f["y_mm"] * ppm)
        hw, hh = int(f["width_mm"] * ppm / 2) + 6, int(f["height_mm"] * ppm / 2) + 6
        cv2.rectangle(out, (cx - hw, cy - hh), (cx + hw, cy + hh), _COL[f["severity"]], 2)
        cv2.putText(out, f"{f.get('type', '')} {f['severity']} {'wide' if f.get('wide') else str(f['area_mm2']) + 'mm2'}".strip(), (cx - hw, max(14, cy - hh - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, _COL[f["severity"]], 1, cv2.LINE_AA)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(path, out)
    return path


def _draw_heatmap(img, hm, threshold, path="results/last_fabric_heatmap.png"):
    """Anomaly score map over the photo. Colours are scaled so the flag threshold falls
    at yellow: yellow-red = above threshold (counted), blue-green = normal."""
    if hm.shape[:2] != img.shape[:2]:
        hm = cv2.resize(hm, (img.shape[1], img.shape[0]))
    norm = np.clip(hm / (1.6 * threshold), 0.0, 1.0)
    colour = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    out = cv2.addWeighted(img, 0.5, colour, 0.5, 0)
    contours, _ = cv2.findContours((hm > threshold).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (255, 255, 255), 1)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(path, out)
    return path


_advisor = None


def get_advisor():
    """One advisor (and one loaded Llama) for the whole process."""
    global _advisor
    if _advisor is None:
        _advisor = FabricAdvisor(make_backend()[0], FAB["sop"])
    return _advisor


_guidance = None


def _advice_payload(rec):
    """Facts for the language model: findings plus likely cause and route from the guidance file."""
    global _guidance
    sop_line = FAB["sop"][rec["verdict"]]
    try:
        if _guidance is None:
            import fabric_actions
            _guidance = fabric_actions.load_guidance()
        return build_user_message(rec["findings"], rec["verdict"], rec["fabric"], _guidance, sop_line)
    except Exception:          # no guidance file: fall back to the short payload
        return fp.build_llm_payload(rec["findings"][:5], rec["verdict"], fabric=rec["fabric"], sop_text=sop_line)


def advise(rec, advisor=None, backend_name="mock"):
    t = time.perf_counter()
    if rec["verdict"] == INVALID:
        adv = {"summary": "Photo not usable - no verdict given.", "risk": "Inspection not performed.",
               "actions": rec["invalid_reasons"] + ["Retake the photo."], "verdict": INVALID,
               "source": "template"}
    elif not CFG.get("llm", {}).get("enabled", True):          # Llama advice switched off in config.yaml
        adv = {"summary": "", "risk": "", "actions": [], "verdict": rec["verdict"], "source": "off"}
    else:
        if advisor is None:
            advisor = get_advisor()
        payload = _advice_payload(rec)
        a, src = advisor.advise(payload, rec["findings"], rec["verdict"])
        adv = {**a.model_dump(), "source": src}
    rec["advice"] = adv
    rec["timings"]["llm"] = round((time.perf_counter() - t) * 1000, 2)
    rec["timings"]["total"] = round(rec["timings"]["to_verdict"] + rec["timings"]["llm"], 2)
    con = _db()
    con.execute("UPDATE fabric_inspections SET advice=?, timings=? WHERE rowid=?",
                (json.dumps(adv), json.dumps(rec["timings"]), rec["id"]))
    con.commit(); con.close()
    return adv


def run_once(fabric=None, image_path=None, verbose=True):
    img = cv2.imread(image_path) if image_path else None
    rec = inspect(fabric, img)
    advise(rec)
    if verbose:
        print(json.dumps({"findings": rec["findings"], "advice": rec["advice"]}, indent=2))
        print(f"Fabric: {rec['fabric']} | verdict: {rec['verdict']}")
        if rec.get("uncalibrated"):
            print("WARNING: px_per_mm is a placeholder - sizes in mm are NOT real until a ruler photo is used.")
        if rec["invalid_reasons"]:
            print("Invalid capture:", "; ".join(rec["invalid_reasons"]))
        print("Stage timings (ms):", rec["timings"])
    return rec


def live(fabric, index=0):
    """Webcam loop: Enter = inspect one frame, q = quit. Keeps the model loaded."""
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    if not cap.isOpened():
        print("Camera not found")
        return
    n = 0
    try:
        while input("Enter = inspect, q = quit: ").strip().lower() != "q":
            for _ in range(5):          # drop stale buffered frames
                cap.read()
            ok, frame = cap.read()
            if not ok:
                print("No frame")
                continue
            n += 1
            os.makedirs("results", exist_ok=True)
            path = f"results/live_{n:03d}.jpg"
            cv2.imwrite(path, frame)
            run_once(fabric, path)
    finally:
        cap.release()


if __name__ == "__main__":
    args = sys.argv[1:]
    fab = args[0] if args else None
    rest = args[1:]
    if "--live" in rest:
        live(fab)
    elif len(rest) > 1:                 # several photos in one process (model stays loaded)
        for pth in rest:
            print("==", pth)
            run_once(fab, pth)
    else:
        run_once(fab, rest[0] if rest else None)
