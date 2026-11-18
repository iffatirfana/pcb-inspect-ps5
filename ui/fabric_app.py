"""Fabric inspection dashboard: local web page, fully offline.

Run from the repo root:   python -m ui.fabric_app      then open http://127.0.0.1:5000
On the Pi, open http://<pi-ip>:5000 from any device on the same network.

Verdict, severity and the action fields come from rules (fabric_profile, fabric_actions).
Llama only words the two short actions. Every inspection is logged to SQLite.
"""
import json
import sqlite3
import csv
import io
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, redirect, render_template_string, request, send_file, url_for

import fabric_actions as fa
import fabric_pipeline as fpipe

app = Flask(__name__)
LOCK = threading.Lock()          # one inspection at a time; the Llama runner is not re-entrant
STATE = {"rec": None, "action": None, "error": None, "guidance": None}
FABRIC = fpipe.FAB["fabric"]
STARTED = time.time()
SESSION = {"id": None}
VERDICTS = ("PASS", "REVIEW", "REWORK", "REJECT")
REASONS = ["Correct call", "Not a defect (lint, fold, shadow)", "Wrong severity", "Wrong zone or size", "Defect missed", "Bad photo or lighting", "Other"]

PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Fabric Inspection Station</title>
<style>
 :root{--bg:#111;--card:#1c1c1c;--line:#333;--mut:#9a9a9a}
 body{font-family:Arial,sans-serif;margin:0;background:var(--bg);color:#eee}
 header{display:flex;flex-wrap:wrap;gap:12px;justify-content:space-between;align-items:center;padding:12px 20px;background:#1c1c1c}
 h1{font-size:18px;margin:0} h3{margin:0 0 8px 0;font-size:15px}
 form.row{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
 button,.btn{font-size:15px;padding:9px 18px;border:0;border-radius:6px;background:#2d7ff9;color:#fff;cursor:pointer}
 button.alt{background:#444} input[type=file]{color:#ccc;max-width:230px}
 .banner{padding:18px 20px;font-size:30px;font-weight:bold;letter-spacing:1px;display:flex;justify-content:space-between;flex-wrap:wrap;gap:8px}
 .banner small{font-size:15px;font-weight:normal}
 .PASS{background:#1f8b3a}.REVIEW{background:#b89b00;color:#111}.REWORK{background:#d98200}.REJECT{background:#c62828}.INVALID_CAPTURE{background:#555}
 main{display:grid;grid-template-columns:minmax(0,3fr) minmax(0,2fr);gap:16px;padding:16px}
 @media(max-width:900px){main{grid-template-columns:1fr}}
 .card{background:var(--card);border-radius:8px;padding:14px;margin-bottom:12px}
 img{width:100%;border-radius:6px}
 table{width:100%;border-collapse:collapse;font-size:14px}
 td,th{text-align:left;padding:6px;border-bottom:1px solid var(--line);vertical-align:top}
 .Low{color:#66bb6a}.Medium{color:#ffa726}.High{color:#ff5252}
 li{margin-bottom:6px}.mut{color:var(--mut);font-size:13px}
 .warn{background:#3a2e00;color:#ffd54f;border-radius:6px;padding:8px 12px;margin-bottom:10px;font-size:14px}
 .err{background:#3a0000;color:#ff8a80;border-radius:6px;padding:8px 12px;margin:12px 16px}
 .tag{display:inline-block;padding:2px 8px;border-radius:10px;background:#333;font-size:12px;margin-left:6px}
 .kv{display:grid;grid-template-columns:130px 1fr;gap:4px 10px;font-size:15px}
 .strip{display:grid;grid-template-columns:repeat(auto-fit,minmax(110px,1fr));gap:10px;padding:12px 16px 0}
 .stat{background:var(--card);border-radius:8px;padding:10px 12px}.stat b{display:block;font-size:22px}.stat span{font-size:12px;color:var(--mut)}
 .qual{font-size:22px;text-align:right}.qual small{display:block;font-size:12px;font-weight:normal;opacity:.85}
 select,input[type=text]{background:#2a2a2a;color:#eee;border:1px solid #444;border-radius:6px;padding:7px;font-size:14px}
 .dev{font-size:12px;color:var(--mut);display:flex;gap:14px;flex-wrap:wrap}
</style></head><body>
<header><div><h1>Fabric Inspection Station &middot; offline &middot; {{ fabric }}</h1>
 <div class="dev"><span>This session: {{ dev.session }}</span><span>Total device use: {{ dev.total }}</span>
 <span>Inspections logged: {{ stats.all }}</span></div></div>
<form class="row" method="post" action="{{ url_for('do_inspect') }}" enctype="multipart/form-data">
 <input type="file" name="photo" accept="image/*">
 <button name="source" value="upload">Inspect photo</button>
 <button name="source" value="camera" class="alt">Capture from camera</button>
</form></header>
{% if error %}<div class="err">{{ error }}</div>{% endif %}
<div class="strip">
 <div class="stat"><b>{{ stats.today }}</b><span>Inspected today</span></div>
 <div class="stat"><b style="color:#66bb6a">{{ stats.PASS }}</b><span>Passed</span></div>
 <div class="stat"><b style="color:#ffd54f">{{ stats.REVIEW + stats.REWORK }}</b><span>Review / rework</span></div>
 <div class="stat"><b style="color:#ff5252">{{ stats.REJECT }}</b><span>Rejected</span></div>
 <div class="stat"><b>{{ stats.rate }}</b><span>Pass rate today</span></div>
</div>
{% if r %}
<div class="banner {{ r.verdict }}"><span>{{ r.verdict.replace('_',' ') }}</span>
 <span class="qual">{% if r.cloth_quality %}Cloth quality {{ r.cloth_quality.score }}/100 &middot; Grade {{ r.cloth_quality.grade }}
  <small>{{ r.cloth_quality.label }}{% if a %} &middot; {{ a.disposition }} &middot; urgency: {{ a.urgency }}{% endif %}</small>
  {% else %}<small>Photo not usable</small>{% endif %}</span></div>
<main>
 <div>
  <div class="card">{% if r.image %}<img src="{{ url_for('image') }}?id={{ r.id }}" alt="Inspected fabric">
   {% else %}<p>No image: {{ r.invalid_reasons|join('; ') }}</p>{% endif %}</div>
  {% if r.heatmap_image %}<div class="card"><h3>Anomaly heatmap <span class="tag">second opinion</span></h3>
   <img src="{{ url_for('heatmap') }}?id={{ r.id }}" alt="Anomaly heatmap">
   <p class="mut">Yellow and red areas look unusual compared with normal fabric (outlined in white). The second opinion flags
    the photo when more than {{ (r.anomaly.limit * 100)|round(1) }}% of it is unusual; this photo: {{ (r.anomaly.fraction * 100)|round(1) }}%.</p></div>{% endif %}
  <div class="card"><h3>Fabric info</h3><div class="kv">
   <b>Fabric</b><span>{{ fabric }}</span><b>Scale</b><span>{{ ppm }} px/mm {% if r.uncalibrated %}(not calibrated){% else %}(calibrated){% endif %}</span>
   <b>Inspected</b><span>{{ r.ts }}</span></div></div>
  <div class="card"><h3>Findings</h3>
   {% if r.findings %}<table><tr><th>Type</th><th>Severity</th><th>Area</th><th>Zone</th><th>Conf.</th></tr>
   {% for f in r.findings %}<tr><td>{{ f.type }}</td><td class="{{ f.severity }}">{{ f.severity }}</td>
    <td>{% if f.wide %}wide area{% else %}{{ f.area_mm2 }} mm&sup2;{% endif %}</td><td>{{ f.zone }}</td><td>{{ f.confidence if f.confidence is not none else '-' }}</td></tr>{% endfor %}</table>
   {% else %}<p class="mut">No defects found.</p>{% endif %}</div>
 </div>
 <div>
  {% for n in (a.notes if a else []) %}<div class="warn">{{ n }}</div>{% endfor %}
  <div class="card"><h3>Recommended action <span class="tag">from rules</span></h3>
   {% if a and a['items'] %}{% for i in a['items'] %}
   <div class="kv" style="margin-bottom:10px">
    <b>Defect</b><span>{{ i.type }} &middot; <span class="{{ i.severity }}">{{ i.severity }}</span> &middot; {{ i.zone }}</span>
    <b>Route</b><span>{{ i.route }}</span><b>Check</b><span>{{ i.check }}</span>
    <b>Likely cause</b><span>{{ i.likely_cause }}</span></div>{% endfor %}
   {% elif a %}<p class="mut">Nothing to route.</p>{% endif %}</div>
  <div class="card">{% if not r.advice or r.advice.source != 'off' %}<h3>Operator actions <span class="tag">{{ r.advice.source if r.advice else '-' }}</span></h3>{% endif %}
   {% if r.advice and r.advice.source != 'off' %}<ol>{% for x in r.advice.actions %}<li>{{ x }}</li>{% endfor %}</ol>
   {% if r.advice.source.startswith('llm') and r.advice.risk %}<p><b>Risk:</b> {{ r.advice.risk }}</p>{% endif %}
   <p class="mut">{{ r.advice.summary }}</p>{% endif %}
   <p class="mut">Detect {{ r.timings.get('detect','-') }} ms &middot; to verdict {{ r.timings.to_verdict }} ms
    &middot; advice {{ r.timings.get('llm','-') }} ms &middot; total {{ r.timings.get('total','-') }} ms</p></div>
  <div class="card"><h3>Operator feedback</h3>
   <form method="post" action="{{ url_for('label') }}" style="display:grid;gap:8px">
    <label class="mut">Correct verdict
     <select name="corrected"><option value="">same as shown</option>{% for v in verdicts %}<option>{{ v }}</option>{% endfor %}</select></label>
    <label class="mut">Reason
     <select name="reason">{% for x in reasons %}<option>{{ x }}</option>{% endfor %}</select></label>
    <input type="text" name="note" maxlength="300" placeholder="Note (optional)">
    <div class="row" style="display:flex;gap:8px;flex-wrap:wrap">
     <button name="kind" value="correct">Correct</button>
     <button name="kind" value="wrong" class="alt">Wrong verdict</button>
     <button name="kind" value="missed" class="alt">Missed defect</button>
     <a class="btn alt" href="{{ url_for('export_csv') }}" style="text-decoration:none">Export CSV</a></div></form></div>
 </div>
</main>
{% else %}<main><div class="card" style="grid-column:1/-1">Upload a photo or press <b>Capture from camera</b>.</div></main>{% endif %}
<div style="padding:0 16px 16px"><div class="card"><h3>Recent inspections</h3><table>
 <tr><th>Time</th><th>Verdict</th><th>Quality</th><th>Defects</th><th>Total ms</th><th>Operator</th></tr>
 {% for h in history %}<tr><td>{{ h.ts }}</td><td>{{ h.verdict }}</td><td>{{ h.q if h.q is not none else '-' }}</td><td>{{ h.n }}</td><td>{{ h.ms }}</td><td>{{ h.label or '' }}</td></tr>{% endfor %}
</table></div></div></body></html>"""


def history(limit=10):
    db = Path(fpipe.CFG["paths"]["db"])
    if not db.exists():
        return []
    con = sqlite3.connect(db)
    try:
        rows = con.execute("SELECT ts, verdict, findings, timings, operator_label, quality FROM fabric_inspections "
                           "ORDER BY rowid DESC LIMIT ?", (limit,)).fetchall()
    except sqlite3.OperationalError:     # table not created yet
        rows = []
    con.close()
    out = []
    for ts, v, fnd, tm, lab, q in rows:
        t = json.loads(tm or "{}")
        out.append({"ts": ts, "verdict": v, "n": len(json.loads(fnd or "[]")),
                    "ms": t.get("total", t.get("to_verdict", "-")), "label": lab,
                    "q": int(q) if q not in (None, "") else None})
    return out


def _hms(seconds):
    s = int(seconds)
    return "%d h %02d min" % (s // 3600, (s % 3600) // 60)


def device_info():
    session = time.time() - STARTED
    total = fpipe.usage_total_seconds()
    if SESSION["id"] is None:            # session row is not written (tests); add this run's time by hand
        total += session
    return {"session": _hms(session), "total": _hms(total)}


def stats():
    out = {"all": 0, "today": 0, "PASS": 0, "REVIEW": 0, "REWORK": 0, "REJECT": 0, "rate": "-"}
    db = Path(fpipe.CFG["paths"]["db"])
    if not db.exists():
        return out
    con = sqlite3.connect(db)
    try:
        out["all"] = con.execute("SELECT COUNT(*) FROM fabric_inspections").fetchone()[0]
        today = time.strftime("%Y-%m-%d")
        for v, n in con.execute("SELECT verdict, COUNT(*) FROM fabric_inspections WHERE ts LIKE ? GROUP BY verdict",
                                (today + "%",)):
            out["today"] += n
            if v in out:
                out[v] = n
    except sqlite3.OperationalError:
        pass
    con.close()
    valid = out["PASS"] + out["REVIEW"] + out["REWORK"] + out["REJECT"]
    if valid:
        out["rate"] = "%d%%" % round(100.0 * out["PASS"] / valid)
    return out


def grab_camera(index=0):
    """One webcam frame (MJPG 1920x1080). Raises RuntimeError if there is no camera."""
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2) if sys.platform.startswith("linux") else cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    try:
        if not cap.isOpened():
            raise RuntimeError("Camera not found")
        for _ in range(5):               # drop stale buffered frames
            cap.read()
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("Camera gave no frame")
        return frame
    finally:
        cap.release()


@app.get("/")
def index():
    return render_template_string(PAGE, r=STATE["rec"], a=STATE["action"], error=STATE["error"],
                                  history=history(), fabric=FABRIC, dev=device_info(), stats=stats(),
                                  ppm=fpipe.FAB["fabrics"][FABRIC]["px_per_mm"], verdicts=VERDICTS, reasons=REASONS)


@app.post("/inspect")
def do_inspect():
    STATE["error"] = None
    try:
        if request.form.get("source") == "camera":
            img = grab_camera(fpipe.CFG.get("camera", {}).get("usb_index", 0))
        else:
            f = request.files.get("photo")
            if not f or not f.filename:
                raise ValueError("Choose a photo first.")
            img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("That file is not a readable image.")
        with LOCK:
            rec = fpipe.inspect(FABRIC, img)
            fpipe.advise(rec)
        if STATE["guidance"] is None:
            STATE["guidance"] = fa.load_guidance()
        STATE["action"] = (None if rec["verdict"] == fpipe.INVALID else
                           fa.structured_action(rec["findings"], rec["verdict"], STATE["guidance"],
                                                rec.get("uncalibrated", False)))
        if rec.get("anomaly", {}).get("flagged") and STATE["action"]:
            STATE["action"]["notes"].insert(0, "Flagged by the second-opinion model only (detector found no defect): "
                                               "confirm by eye.")
        STATE["rec"] = rec
    except Exception as e:               # show the problem on the page instead of a 500
        STATE["error"] = str(e)
    return redirect(url_for("index"))


@app.post("/label")
def label():
    rec = STATE["rec"]
    kind = request.form.get("kind")
    if rec and kind in ("correct", "wrong", "missed"):
        corrected = request.form.get("corrected")
        fpipe.operator_label(rec["id"], kind, request.form.get("reason"), request.form.get("note"),
                             corrected if corrected in VERDICTS else None)
    return redirect(url_for("index"))


@app.get("/export.csv")
def export_csv():
    con = sqlite3.connect(fpipe.CFG["paths"]["db"])
    try:
        rows = con.execute("SELECT ts, fabric, verdict, quality, operator_label, corrected, reason, note "
                           "FROM fabric_inspections ORDER BY rowid").fetchall()
    except sqlite3.OperationalError:
        rows = []
    con.close()
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["time", "fabric", "verdict", "quality", "operator_label", "corrected_verdict", "reason", "note"])
    w.writerows(rows)
    return Response(buf.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=fabric_inspections.csv"})


@app.get("/image")
def image():
    rec = STATE["rec"]
    if not rec or not rec.get("image"):
        return ("", 404)
    resp = send_file(Path(rec["image"]).resolve(), mimetype="image/png")
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.get("/heatmap")
def heatmap():
    rec = STATE["rec"]
    if not rec or not rec.get("heatmap_image"):
        return ("", 404)
    resp = send_file(Path(rec["heatmap_image"]).resolve(), mimetype="image/png")
    resp.headers["Cache-Control"] = "no-store"
    return resp


def _warm():
    """Load the detector and the Llama runner once at start-up so the first click is fast."""
    try:
        with LOCK:
            fpipe.get_detector(FABRIC)
            adv = fpipe.get_advisor()
            adv.backend.generate(adv.build_prompt("Fabric: warm-up\nVerdict: REVIEW"))
    except Exception:
        pass                              # first real inspection will report any problem


def _usage_loop():
    SESSION["id"] = fpipe.usage_start()
    while True:
        time.sleep(30)
        fpipe.usage_tick(SESSION["id"], time.time() - STARTED)


if __name__ == "__main__":
    threading.Thread(target=_usage_loop, daemon=True).start()
    threading.Thread(target=_warm, daemon=True).start()
    app.run(host="0.0.0.0", port=5000, threaded=True)
