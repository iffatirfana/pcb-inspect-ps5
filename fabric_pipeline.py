"""Fabric inspection pipeline (separate from the PCB pipeline.py; shares camera, quality gate, LLM backend, DB).

  capture -> quality gate -> crop -> anomaly heatmap -> findings (mm) -> rules -> VERDICT
          -> Llama advice (afterwards; never changes the verdict) -> SQLite log

Run:  python fabric_pipeline.py <fabric> [image.png]     (fabric: white | purple)
"""
import json
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import yaml

import fabric_profile as fp
from fabric_advisor import FabricAdvisor, FabricMockBackend
from vision.quality import check_capture

CFG = yaml.safe_load(open("config.yaml", encoding="utf-8"))
FAB = yaml.safe_load(open("kb/fabric.yaml", encoding="utf-8"))
INVALID = "INVALID_CAPTURE"
_scorers = {}


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
    return con


def log(rec):
    con = _db()
    cur = con.execute("INSERT INTO fabric_inspections VALUES (?,?,?,?,?,?,NULL)",
                      (rec["ts"], rec["fabric"], rec["verdict"], json.dumps(rec["findings"]),
                       json.dumps(rec.get("advice")), json.dumps(rec["timings"])))
    con.commit(); rid = cur.lastrowid; con.close()
    return rid


def operator_label(inspection_id, label):
    """label: correct | wrong | missed (operator tick/cross)."""
    con = _db()
    con.execute("UPDATE fabric_inspections SET operator_label=? WHERE rowid=?", (label, inspection_id))
    con.commit(); con.close()


def inspect(fabric=None, image=None, scorer=None):
    """Everything up to the verdict. Returns the record (already logged)."""
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
        hm = (scorer or get_scorer(fabric)).heatmap(img);       lap("heatmap")
        findings = fp.findings_from_heatmap(hm, c["threshold"], c["px_per_mm"], rules)
        rec["findings"] = findings
        rec["verdict"] = fp.verdict_from_findings(findings, rules);  lap("rules")
        rec["image"] = _draw(img, findings, c["px_per_mm"]);    lap("draw")
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
        cv2.putText(out, f"{f['severity']} {f['area_mm2']}mm2", (cx - hw, max(14, cy - hh - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, _COL[f["severity"]], 1, cv2.LINE_AA)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(path, out)
    return path


def advise(rec, advisor=None, backend_name="mock"):
    t = time.perf_counter()
    if rec["verdict"] == INVALID:
        adv = {"summary": "Photo not usable - no verdict given.", "risk": "Inspection not performed.",
               "actions": rec["invalid_reasons"] + ["Retake the photo."], "verdict": INVALID,
               "source": "template"}
    else:
        if advisor is None:
            advisor = FabricAdvisor(make_backend()[0], FAB["sop"])
        payload = fp.build_llm_payload(rec["findings"][:5], rec["verdict"], fabric=rec["fabric"],
                                       sop_text=FAB["sop"][rec["verdict"]])
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
        if rec["invalid_reasons"]:
            print("Invalid capture:", "; ".join(rec["invalid_reasons"]))
        print("Stage timings (ms):", rec["timings"])
    return rec


if __name__ == "__main__":
    run_once(sys.argv[1] if len(sys.argv) > 1 else None, sys.argv[2] if len(sys.argv) > 2 else None)
