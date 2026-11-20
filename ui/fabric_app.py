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
import fabric_quality as fq
import fabric_pattern as fpat
import fabric_calib as fcal
import fabric_pipeline as fpipe
from ui.live_camera import LiveCamera

app = Flask(__name__)
LOCK = threading.Lock()          # one inspection at a time; the Llama runner is not re-entrant
STATE = {"pending": None, "rec": None, "action": None, "error": None, "guidance": None, "label": None, "fabric": None}
PSTATE = {"result": None}
FABRIC = fpipe.FAB["fabric"]
STARTED = time.time()
SESSION = {"id": None}
LIVE = LiveCamera(index=fpipe.CFG.get("camera", {}).get("usb_index", 0))
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
 .why{border-left:4px solid #2d7ff9;padding:8px 12px;background:#202833;border-radius:4px;font-size:15px}
 .legend{height:12px;border-radius:6px;background:linear-gradient(to right,#00007f,#0000ff,#00ffff,#ffff00,#ff0000,#7f0000);position:relative;margin:10px 0 4px}
 .legend i{position:absolute;top:-4px;height:20px;border-left:2px solid #fff}
 .lg{display:flex;justify-content:space-between;font-size:12px;color:var(--mut)}
 tr.over td{background:#3a1f1f} details summary{cursor:pointer;color:#9ecbff;font-size:14px} details p,details li{font-size:13px;color:#ccc}
 .badge{display:inline-block;padding:3px 10px;border-radius:12px;font-size:13px;font-weight:bold;margin-left:8px}.ai{background:#333;color:#ddd}.op{background:#1f6b3a;color:#fff}.opx{background:#8a4b00;color:#fff}
 .dev{font-size:12px;color:var(--mut);display:flex;gap:14px;flex-wrap:wrap}
</style></head><body>
<header><div><h1>Fabric Inspection Station &middot; offline &middot; {{ fabric }}</h1>
 <div class="dev"><span>This session: {{ dev.session }}</span><span>Total device use: {{ dev.total }}</span>
 <span>Inspections logged: {{ stats.all }}</span></div></div>
<form class="row" method="post" action="{{ url_for('set_fabric') }}">
 <select name="fabric" onchange="this.form.submit()" title="Fabric">{% for k in fabrics %}<option {{ 'selected' if k == fabric else '' }}>{{ k }}</option>{% endfor %}</select>
 <a class="btn alt" href="{{ url_for('pattern_page') }}" style="text-decoration:none">Pattern check</a>
</form>
<form class="row" method="post" action="{{ url_for('select_area') }}" enctype="multipart/form-data">
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
<div style="padding:12px 16px 0"><div class="card"><h3>Live camera
  <span class="tag">{% if live.running %}on{% else %}off{% endif %}</span>
  {% if live.running and live.size %}<span class="tag">{{ live.size[0] }} &times; {{ live.size[1] }}</span>{% endif %}</h3>
 {% if live.running %}
  <img src="{{ url_for('live_stream') }}" alt="Live camera" style="max-width:760px;width:100%">
  <div class="row" style="display:flex;gap:8px;margin-top:10px;flex-wrap:wrap">
   <form method="post" action="{{ url_for('select_area') }}"><button name="source" value="live">Capture &amp; select area</button></form>
   <form method="post" action="{{ url_for('live_stop') }}"><button class="alt">Stop camera</button></form></div>
  <p class="mut">The photo that is inspected is the frame you see here, at full camera size.</p>
 {% else %}
  <form method="post" action="{{ url_for('live_start') }}"><button>Start live view</button></form>
  <p class="mut">Shows the camera here so you can place the cloth, then capture and inspect with one click.
   Press Stop afterwards so other tools can use the camera.</p>
 {% endif %}
 {% if live.error %}<div class="err" style="margin:8px 0 0">{{ live.error }}</div>{% endif %}</div></div>
{% if r %}
<div class="banner {{ r.verdict }}"><span>{{ r.verdict.replace('_',' ') }}
  {% if lab %}{% if lab.corrected %}<span class="badge opx">Operator changed to {{ lab.corrected }}</span>
  {% elif lab.kind == 'correct' %}<span class="badge op">Operator confirmed</span>
  {% else %}<span class="badge opx">Operator: {{ lab.kind }}</span>{% endif %}
  {% else %}<span class="badge ai">AI verdict &middot; not yet confirmed</span>{% endif %}
  <small style="display:block;margin-top:4px">Inspection #{{ r.id }} &middot; {{ r.ts }}</small></span>
 <span class="qual">{% if r.cloth_quality %}Cloth quality {{ r.cloth_quality.score }}/100 &middot; Grade {{ r.cloth_quality.grade }}
  <small>{{ r.cloth_quality.label }}{% if a %} &middot; {{ a.disposition }} &middot; urgency: {{ a.urgency }}{% endif %}</small>
  {% else %}<small>Photo not usable</small>{% endif %}</span></div>
<main>
 <div>
  <div class="card why"><b>Why this verdict:</b> {{ why }}</div>
  <div class="card">{% if r.image %}<img src="{{ url_for('image') }}?id={{ r.id }}" alt="Inspected fabric">
   {% else %}<p>No image: {{ r.invalid_reasons|join('; ') }}</p>{% endif %}</div>
  {% if r.heatmap_image %}<div class="card"><h3>Anomaly heatmap <span class="tag">second opinion</span></h3>
   <img src="{{ url_for('heatmap') }}?id={{ r.id }}" alt="Anomaly heatmap">
   <div class="legend"><i style="left:62.5%"></i></div>
   <div class="lg"><span>0 &middot; normal</span><span>flag threshold</span><span>high &middot; unusual</span></div>
   <p class="mut">Real output of the second-opinion model (ResNet18 features compared with a bank of normal denim), not a drawing.</p>
   <p class="mut">Yellow and red areas look unusual compared with normal fabric (outlined in white). The second opinion flags
    the photo when more than {{ (r.anomaly.limit * 100)|round(1) }}% of it is unusual; this photo: {{ (r.anomaly.fraction * 100)|round(1) }}%.</p></div>{% endif %}
  <div class="card"><h3>Fabric and measurement</h3><div class="kv">
   <b>Fabric</b><span>{{ fabric }}</span>
   <b>Scale</b><span>{{ ppm }} px/mm &middot; {% if r.calibration_applied is defined and not r.calibration_applied %}uploaded photo, not straightened: sizes are estimates{% elif r.uncalibrated %}not calibrated{% else %}calibrated (ruler photo){% endif %}</span>
   <b>Camera frame</b><span>{% if r.input_size %}{{ r.input_size[0] }} &times; {{ r.input_size[1] }} px{% else %}-{% endif %}</span>
   <b>Area inspected</b><span>{% if r.crop %}{{ r.crop.size[0] }} &times; {{ r.crop.size[1] }} px, at x {{ r.crop.px[0] }}, y {{ r.crop.px[1] }}{% if r.crop.small %} (small area){% else %}{% endif %}{% else %}whole photo{% endif %}</span>
   <b>Model</b><span>{{ info.model }} &middot; confidence &ge; {{ info.conf }}</span>
   <b>Inference</b><span>{{ r.timings.get('detect', '-') }} ms</span>
   <b>Units</b><span>area in mm&sup2;, width and height in mm</span></div></div>
  <div class="card"><h3>Findings{% if counts %} <span class="tag">{{ counts }}</span>{% endif %}</h3>
   {% if r.findings %}<table><tr><th>#</th><th>Type</th><th>Severity</th><th>Area (mm&sup2;)</th><th>W &times; H (mm)</th><th>Zone</th><th>Conf.</th></tr>
   {% for f in r.findings %}<tr class="{{ 'over' if f.severity != 'Low' else '' }}"><td>{{ loop.index }}</td><td>{{ f.type }}</td><td class="{{ f.severity }}">{{ f.severity }}</td>
    <td>{% if f.wide %}wide area{% elif f.area_mm2 %}{{ f.area_mm2 }}{% else %}-{% endif %}</td>
    <td>{% if f.width_mm and f.height_mm %}{{ f.width_mm }} &times; {{ f.height_mm }}{% else %}-{% endif %}</td>
    <td>{{ f.zone }}</td><td>{{ (f.confidence * 100)|round|int ~ '%' if f.confidence is not none else '-' }}</td></tr>{% endfor %}</table>
   <p class="mut">Highlighted rows are Medium or High: above the {{ rules.low_max_mm2 }} mm&sup2; limit for a small mark. Size comes from the calibrated scale.</p>
   {% else %}<p class="mut">No defects found.</p>{% endif %}</div>
 </div>
 <div>
  {% for n in (a.notes if a else []) %}<div class="warn">{{ n }}</div>{% endfor %}
  <div class="card"><h3>Recommended action</h3>
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
  <div class="card"><details><summary>How the quality score and grade are worked out</summary>
   <p>Start at 100. Each finding costs points by severity: Low {{ q.Low }}, Medium {{ q.Medium }}, High {{ q.High }},
    up to 50% more for large defects (area up to 400 mm&sup2;). A photo flagged only by the second-opinion model costs {{ q.anom }}.</p>
   <p>The verdict sets a ceiling so the grade never contradicts it: PASS up to 100, REVIEW {{ q.REVIEW }}, REWORK {{ q.REWORK }}, REJECT {{ q.REJECT }}.</p>
   <p>Grade A = 90 or more, B = 75-89, C = 55-74, D = below 55.
    Severity: Low is under {{ rules.low_max_mm2 }} mm&sup2;, Medium is {{ rules.low_max_mm2 }} to {{ rules.medium_max_mm2 }} mm&sup2;, High is {{ rules.medium_max_mm2 }} mm&sup2; or more.</p>
   <p>The verdict is set by rules, not by a language model. Findings are AI detections until an operator confirms them below.</p></details></div>
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
<div style="padding:0 16px 16px"><div class="card"><h3>Recent inspections <a href="{{ url_for('log_page') }}" style="font-size:13px;color:#8ab4f8;margin-left:10px">Full log</a></h3><p class="mut">Click a time to open that inspection's report.</p><table>
 <tr><th>Time</th><th>Verdict</th><th>Quality</th><th>Defects</th><th>Total ms</th><th>Operator</th></tr>
 {% for h in history %}<tr><td><a href="{{ url_for('report', rid=h.id) }}" style="color:#8ab4f8">{{ h.ts }}</a></td><td>{{ h.verdict }}</td><td>{{ h.q if h.q is not none else '-' }}</td><td>{{ h.n }}</td><td>{{ h.ms }}</td><td>{{ h.label or '' }}</td></tr>{% endfor %}
</table></div></div></body></html>"""


def fab():
    """Fabric chosen in the header (defaults to the one in kb/fabric.yaml)."""
    return STATE["fabric"] or FABRIC


def fabric_choices():
    return [k for k, v in fpipe.FAB["fabrics"].items() if v.get("yolo_model")]


def history(limit=10):
    db = Path(fpipe.CFG["paths"]["db"])
    if not db.exists():
        return []
    con = sqlite3.connect(db)
    try:
        rows = con.execute("SELECT rowid, ts, verdict, findings, timings, operator_label, quality FROM fabric_inspections "
                           "ORDER BY rowid DESC LIMIT ?", (limit,)).fetchall()
    except sqlite3.OperationalError:     # table not created yet
        rows = []
    con.close()
    out = []
    for rid, ts, v, fnd, tm, lab, q in rows:
        t = json.loads(tm or "{}")
        out.append({"id": rid, "ts": ts, "verdict": v, "n": len(json.loads(fnd or "[]")),
                    "ms": t.get("total", t.get("to_verdict", "-")), "label": lab,
                    "q": int(q) if q not in (None, "") else None})
    return out


def why_verdict(rec):
    """One plain sentence, from the same rules that set the verdict."""
    rules = fpipe.FAB["rules"]
    f = rec.get("findings") or []
    if rec["verdict"] == fpipe.INVALID:
        return "The photo failed the quality gate (%s), so nothing was judged." % ("; ".join(rec["invalid_reasons"]) or "unusable")
    if not f:
        return "No defect was found, so the piece passes."
    if all(x.get("type") == "Anomaly" for x in f):
        a = rec.get("anomaly") or {}
        return ("Only the second-opinion model flagged this photo (%.1f%% of it looks unusual, limit %.1f%%). "
                "That can raise PASS to REVIEW, never more: check it by eye." % (100 * a.get("fraction", 0), 100 * a.get("limit", 0)))
    if all(x.get("type") == "Pattern gap" for x in f):
        return ("%d place(s) where a printed motif seems to be missing. The pattern check is a hint only: "
                "REVIEW, never rework or reject. Check the marked spot by eye." % len(f))
    big = max(f, key=lambda x: x.get("area_mm2") or 0)
    sev = {x["severity"] for x in f}
    size = "%s of %s mm\u00b2" % (big.get("type", "Defect"), big.get("area_mm2"))
    if "High" in sev:
        return "REJECT because the largest defect (%s) is at least %s mm\u00b2." % (size, rules["medium_max_mm2"])
    if "Medium" in sev:
        return "REWORK because the largest defect (%s) is between %s and %s mm\u00b2." % (size, rules["low_max_mm2"], rules["medium_max_mm2"])
    if len(f) > rules["max_defects_for_review"]:
        return "REWORK because there are %d defects, more than the %d allowed for a review." % (len(f), rules["max_defects_for_review"])
    return "REVIEW because only small marks were found (largest %s, under %s mm\u00b2): an operator checks by eye." % (size, rules["low_max_mm2"])


def category_counts(rec):
    c = {}
    for x in rec.get("findings") or []:
        c[x.get("type", "Defect")] = c.get(x.get("type", "Defect"), 0) + 1
    return ", ".join("%s \u00d7%d" % (k, v) for k, v in c.items())


def model_info():
    c = fpipe.FAB["fabrics"][fab()]
    return {"model": Path(c.get("yolo_model", c.get("backbone_onnx", "-"))).name, "conf": c.get("yolo_conf", "-")}


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
                                  history=history(), fabric=fab(), fabrics=fabric_choices(), live=LIVE, dev=device_info(), stats=stats(),
                                  ppm=fpipe.FAB["fabrics"][fab()]["px_per_mm"], verdicts=VERDICTS, reasons=REASONS,
                                  lab=STATE["label"], rules=fpipe.FAB["rules"], info=model_info(),
                                  why=(why_verdict(STATE["rec"]) if STATE["rec"] else ""),
                                  counts=(category_counts(STATE["rec"]) if STATE["rec"] else ""),
                                  q={"Low": fq.SEVERITY_COST["Low"], "Medium": fq.SEVERITY_COST["Medium"], "High": fq.SEVERITY_COST["High"],
                                     "anom": fq.ANOMALY_COST, "REVIEW": fq.CEILING["REVIEW"], "REWORK": fq.CEILING["REWORK"],
                                     "REJECT": fq.CEILING["REJECT"]})


def _read_source():
    """The photo for this request: live frame, USB camera, or uploaded file."""
    if request.form.get("source") == "live" or (request.form.get("source") == "camera" and LIVE.running):
        img = LIVE.snapshot()
        if img is None:
            raise ValueError("Start the live view and wait for the picture first.")
    elif request.form.get("source") == "camera":
        img = grab_camera(fpipe.CFG.get("camera", {}).get("usb_index", 0))
    else:
        f = request.files.get("photo")
        if not f or not f.filename:
            raise ValueError("Choose a photo first.")
        img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("That file is not a readable image.")
    return img


def _from_camera():
    """Only pictures taken by the calibrated camera are straightened; uploaded photos are used as they are."""
    return request.form.get("source") == "live" or request.form.get("source") == "camera"


def _run_inspection(img, crop_frac=None, calibrate=True):
    with LOCK:
        rec = fpipe.inspect(fab(), img, crop_frac=crop_frac, calibrate=calibrate)
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
    STATE["label"] = None


@app.post("/inspect")
def do_inspect():
    STATE["error"] = None
    try:
        _run_inspection(_read_source(), calibrate=_from_camera())
    except Exception as e:               # show the problem on the page instead of a 500
        STATE["error"] = str(e)
    return redirect(url_for("index"))


def _preview(img):
    """The picture the rectangle is drawn on: straightened (camera pictures only) and cut like the pipeline does,
    before the operator's crop."""
    c = fpipe.FAB["fabrics"][fab()]
    return fpipe.fp.crop_image(fcal.apply(img, c) if STATE.get("pending_cal", True) else img, c.get("crop"))


@app.post("/select")
def select_area():
    """Step 1 of every inspection: take the photo, then let the operator draw the area to check."""
    STATE["error"] = None
    try:
        STATE["pending"] = _read_source()
        STATE["pending_cal"] = _from_camera()
    except Exception as e:
        STATE["error"] = str(e)
        return redirect(url_for("index"))
    h, w = _preview(STATE["pending"]).shape[:2]
    return render_template_string(SELECT_PAGE, fabric=fab(), w=w, h=h, error=None, t=int(time.time()))


@app.get("/select/img")
def select_img():
    if STATE["pending"] is None:
        return Response(status=404)
    ok, buf = cv2.imencode(".jpg", _fit(_preview(STATE["pending"]), 1400), [cv2.IMWRITE_JPEG_QUALITY, 88])
    return Response(buf.tobytes(), mimetype="image/jpeg", headers={"Cache-Control": "no-store"})


@app.post("/inspect/area")
def inspect_area():
    STATE["error"] = None
    img = STATE["pending"]
    try:
        if img is None:
            raise ValueError("No photo waiting. Choose or capture a photo first.")
        frac = None
        if request.form.get("whole") is None:
            vals = [request.form.get(k, "") for k in ("fx", "fy", "fw", "fh")]
            if not all(vals):
                raise ValueError("Draw a rectangle on the photo first, or press Inspect whole photo.")
            frac = tuple(float(v) for v in vals)
        _run_inspection(img, frac, calibrate=STATE.get("pending_cal", True))
        STATE["pending"] = None
    except Exception as e:
        STATE["error"] = str(e)
    return redirect(url_for("index"))


SELECT_PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Select area</title>
<style>body{background:#111;color:#eee;font-family:system-ui,Segoe UI,Arial;margin:0;padding:16px}
.btn,button{background:#2e7d32;color:#fff;border:0;border-radius:6px;padding:10px 16px;font-size:15px;cursor:pointer}
button:disabled{background:#333;color:#888;cursor:not-allowed}.alt{background:#444}
#wrap{position:relative;display:inline-block;touch-action:none;user-select:none}
#img{display:block;max-width:100%;height:auto;max-height:70vh;pointer-events:none;-webkit-user-drag:none;user-select:none}#cv{max-width:100%;max-height:70vh;border:2px solid #ffd54f}
#box{position:absolute;border:2px solid #ffd54f;background:rgba(255,213,79,.15);display:none;pointer-events:none}
.mut{color:#aaa;font-size:13px}.bar{margin:0 0 12px;display:flex;gap:8px;flex-wrap:wrap;align-items:center}</style></head><body>
<h2>Inspect &middot; {{ fabric }}</h2>
<p id="msg" class="mut">Step 1 of 3: choose how to inspect this photo.</p>
<form method="post" action="{{ url_for('inspect_area') }}">
 <input type="hidden" name="fx"><input type="hidden" name="fy"><input type="hidden" name="fw"><input type="hidden" name="fh">
 <div class="bar" id="s1"><button type="button" id="bcrop">Crop an area</button>
  <button class="alt" name="whole" value="1">Inspect full photo</button>
  <a class="btn alt" href="{{ url_for('index') }}" style="text-decoration:none">Cancel</a>
  <span class="mut">Photo {{ w }} &times; {{ h }} px</span></div>
 <div class="bar" id="s2" style="display:none"><button type="button" id="bdone" disabled>Crop</button>
  <button type="button" class="alt" id="bback">Back</button><span id="info" class="mut">Drag a rectangle on the photo.</span></div>
 <div class="bar" id="s3" style="display:none"><button id="bgo">Inspect this part</button>
  <button type="button" class="alt" id="bredo">Crop again</button><span id="info3" class="mut"></span></div>
</form>
<div id="wrap"><img id="img" draggable="false" src="{{ url_for('select_img') }}?t={{ t }}" alt="photo"><div id="box"></div></div>
<canvas id="cv" style="display:none"></canvas>

<script>
const $=id=>document.getElementById(id),img=$('img'),box=$('box'),wrap=$('wrap'),cv=$('cv'),msg=$('msg');
const f=n=>document.getElementsByName(n)[0],W={{ w }},H={{ h }};let mode='choose',s=null,ok=false;
function show(m){mode=m;$('s1').style.display=m=='choose'?'flex':'none';$('s2').style.display=m=='draw'?'flex':'none';
 $('s3').style.display=m=='preview'?'flex':'none';wrap.style.display=m=='preview'?'none':'inline-block';cv.style.display=m=='preview'?'block':'none';
 box.style.display=(m=='draw'&&s!==null||m=='draw'&&ok)?'block':'none';
 msg.textContent={choose:'Step 1 of 3: choose how to inspect this photo.',draw:'Step 2 of 3: drag a rectangle over the part to check. It is cropped as soon as you let go.',
  preview:'Step 3 of 3: this is the cropped part. Press Inspect this part, or crop again.'}[m]}
function pt(e){const r=img.getBoundingClientRect();return[Math.min(r.width,Math.max(0,e.clientX-r.left)),Math.min(r.height,Math.max(0,e.clientY-r.top)),r]}
$('bcrop').onclick=()=>{ok=false;box.style.display='none';$('bdone').disabled=true;$('info').textContent='Drag a rectangle on the photo.';show('draw')};
$('bback').onclick=()=>{ok=false;box.style.display='none';show('choose')};
$('bredo').onclick=()=>{$('bcrop').onclick()};
wrap.addEventListener('pointerdown',e=>{if(mode!='draw')return;const[x,y]=pt(e);s=[x,y];wrap.setPointerCapture(e.pointerId);box.style.display='block';draw(x,y)});
wrap.addEventListener('pointermove',e=>{if(s&&mode=='draw'){const[x,y]=pt(e);draw(x,y)}});
wrap.addEventListener('pointerup',e=>{if(s&&mode=='draw'){const[x,y]=pt(e);draw(x,y);s=null;if(ok)$('bdone').onclick()}s=null});wrap.addEventListener('pointercancel',e=>{s=null});wrap.addEventListener('dragstart',e=>e.preventDefault());wrap.addEventListener('contextmenu',e=>e.preventDefault());
function draw(x,y){const r=img.getBoundingClientRect(),l=Math.min(s[0],x),t=Math.min(s[1],y),w=Math.abs(x-s[0]),h=Math.abs(y-s[1]);
 Object.assign(box.style,{left:l+'px',top:t+'px',width:w+'px',height:h+'px'});
 f('fx').value=(l/r.width).toFixed(4);f('fy').value=(t/r.height).toFixed(4);f('fw').value=(w/r.width).toFixed(4);f('fh').value=(h/r.height).toFixed(4);
 const pw=Math.round(w/r.width*W),ph=Math.round(h/r.height*H);ok=pw>=64&&ph>=64;$('bdone').disabled=!ok;
 $('info').textContent='Selected '+pw+' \u00d7 '+ph+' px'+(ok?'':' (too small, minimum 64 px)')}
$('bdone').onclick=()=>{if(!ok)return;const nw=img.naturalWidth,nh=img.naturalHeight,fx=+f('fx').value,fy=+f('fy').value,fw=+f('fw').value,fh=+f('fh').value;
 cv.width=Math.round(fw*nw);cv.height=Math.round(fh*nh);cv.getContext('2d').drawImage(img,fx*nw,fy*nh,fw*nw,fh*nh,0,0,cv.width,cv.height);
 const pw=Math.round(fw*W),ph=Math.round(fh*H);$('info3').textContent='Cropped part: '+pw+' \u00d7 '+ph+' px'+((pw<W/2||ph<H/2)?' (small area: detection may be less reliable)':'');show('preview')};
</script></body></html>"""


# ------------------------------------------------------------------ report for one logged inspection

def severity_reason(f, rules):
    lo, hi = rules["low_max_mm2"], rules["medium_max_mm2"]
    a = f.get("area_mm2")
    if a is None:
        return ""
    if f["severity"] == "Low":
        return "area %s mm\u00b2 is under %s mm\u00b2" % (a, lo)
    if f["severity"] == "Medium":
        return "area %s mm\u00b2 is between %s and %s mm\u00b2" % (a, lo, hi)
    return "area %s mm\u00b2 is %s mm\u00b2 or more" % (a, hi)


def defect_summary(n, f, rules, guidance):
    """One plain summary per logged defect: what, how big, where, how sure, why that severity, what to do."""
    t = f.get("type", "Defect")
    g = guidance["defects"].get(t, guidance["default"])
    where = "%s of the area inspected (about %s mm across, %s mm down)" % (f.get("zone", "-"), f.get("x_mm", "-"), f.get("y_mm", "-"))
    if t == "Pattern gap":
        size = "a place where a printed motif seems to be missing"
        sev = "Pattern gaps are a hint only: they can raise PASS to REVIEW, never more."
    elif t == "Anomaly":
        size = "an area the second-opinion model finds unusual"
        sev = "The second opinion can raise PASS to REVIEW, never more."
    else:
        dims = ""
        if f.get("width_mm") is not None:
            dims = ", %s \u00d7 %s mm" % (f["width_mm"], f["height_mm"])
        size = "%s mm\u00b2%s" % (f.get("area_mm2", "-"), dims)
        sev = "%s severity because the %s." % (f["severity"], severity_reason(f, rules))
    conf = ("%d%%" % round(100 * f["confidence"])) if f.get("confidence") is not None else "not given"
    return {"n": n, "type": t, "severity": f["severity"], "size": size, "where": where, "confidence": conf,
            "why": sev, "check": g["check"], "cause": g["likely_cause"],
            "route": g["route"].get(f["severity"], guidance["default"]["route"][f["severity"]])}


def load_inspection(rid):
    db = Path(fpipe.CFG["paths"]["db"])
    if not db.exists():
        return None
    con = sqlite3.connect(db)
    try:
        row = con.execute("SELECT rowid, ts, fabric, verdict, findings, advice, timings, quality, operator_label, reason, note, "
                          "corrected, extra FROM fabric_inspections WHERE rowid=?", (rid,)).fetchone()
    except sqlite3.OperationalError:
        row = None
    con.close()
    if not row:
        return None
    k = ("id", "ts", "fabric", "verdict", "findings", "advice", "timings", "quality", "label", "reason", "note", "corrected", "extra")
    d = dict(zip(k, row))
    for key in ("findings", "advice", "timings", "extra"):
        d[key] = json.loads(d[key]) if d[key] else ({} if key != "findings" else [])
    d["findings"] = d["findings"] or []
    return d


@app.get("/report/<int:rid>")
def report(rid):
    d = load_inspection(rid)
    if d is None:
        return Response("No such inspection.", status=404)
    if STATE["guidance"] is None:
        STATE["guidance"] = fa.load_guidance()
    rules = fpipe.FAB["rules"]
    ex = d["extra"] or {}
    rec = {"verdict": d["verdict"], "findings": d["findings"], "anomaly": ex.get("anomaly"),
           "invalid_reasons": ex.get("invalid_reasons") or []}
    why = why_verdict(rec)
    defects = [defect_summary(i, f, rules, STATE["guidance"]) for i, f in enumerate(d["findings"], 1)]
    action = None
    if d["verdict"] != fpipe.INVALID:
        action = fa.structured_action(d["findings"], d["verdict"], STATE["guidance"], bool(ex.get("uncalibrated")))
    rd = fpipe.REPORT_DIR
    return render_template_string(REPORT_PAGE, d=d, why=why, defects=defects, action=action, ex=ex,
                                  counts=category_counts(rec), has_img=(rd / ("%d.png" % rid)).exists(),
                                  has_hm=(rd / ("%d_heatmap.png" % rid)).exists(), ppm=fpipe.FAB["fabrics"].get(d["fabric"], {}).get("px_per_mm", "-"))


@app.get("/report/<int:rid>/<kind>.png")
def report_img(rid, kind):
    name = "%d.png" % rid if kind == "picture" else "%d_heatmap.png" % rid
    p = (fpipe.REPORT_DIR / name).resolve()
    if kind not in ("picture", "heatmap") or not p.exists():
        return Response(status=404)
    return send_file(p, mimetype="image/png")


@app.get("/log")
def log_page():
    return render_template_string(LOG_PAGE, rows=history(300))


LOG_PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Inspection log</title>
<style>body{background:#111;color:#eee;font-family:system-ui,Segoe UI,Arial;margin:0;padding:16px}a{color:#8ab4f8}
table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid #333;padding:6px 10px;text-align:left}</style></head><body>
<p><a href="{{ url_for('index') }}">&larr; Dashboard</a></p><h2>Inspection log</h2>
<table><tr><th>#</th><th>Time</th><th>Verdict</th><th>Quality</th><th>Defects</th><th>Total ms</th><th>Operator</th></tr>
{% for h in rows %}<tr><td>{{ h.id }}</td><td><a href="{{ url_for('report', rid=h.id) }}">{{ h.ts }}</a></td><td>{{ h.verdict }}</td>
<td>{{ h.q if h.q is not none else '-' }}</td><td>{{ h.n }}</td><td>{{ h.ms }}</td><td>{{ h.label or '' }}</td></tr>{% endfor %}
</table></body></html>"""

REPORT_PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Inspection report #{{ d.id }}</title>
<style>body{background:#111;color:#eee;font-family:system-ui,Segoe UI,Arial;margin:0;padding:16px;max-width:1100px}a{color:#8ab4f8}
.card{background:#1b1b1b;border-radius:8px;padding:14px;margin:12px 0}h2,h3{margin:0 0 8px}.mut{color:#aaa;font-size:13px}
.v{display:inline-block;padding:4px 14px;border-radius:6px;font-weight:bold;font-size:18px}
.PASS{background:#1f6b3a}.REVIEW{background:#8a6d00}.REWORK{background:#a35200}.REJECT{background:#a11}.INVALID{background:#555}
.kv{display:grid;grid-template-columns:170px 1fr;gap:4px 12px}img{max-width:100%;border-radius:6px}
.d{border-left:4px solid #666;padding:6px 12px;margin:10px 0}.Low{border-color:#0c0}.Medium{border-color:#ff8c00}.High{border-color:#f33}
button{background:#444;color:#fff;border:0;border-radius:6px;padding:8px 14px;cursor:pointer}
@media print{body{background:#fff;color:#000}.card{background:#fff;border:1px solid #999}.noprint{display:none}a{color:#000}}</style></head><body>
<p class="noprint"><a href="{{ url_for('index') }}">&larr; Dashboard</a> &middot; <a href="{{ url_for('log_page') }}">Full log</a>
 &middot; <button onclick="window.print()">Print / save as PDF</button></p>
<h2>Inspection report #{{ d.id }}</h2>
<div class="card"><span class="v {{ d.verdict }}">{{ d.verdict }}</span> <span class="mut">&nbsp;{{ d.ts }} &middot; fabric {{ d.fabric }}
 {% if d.quality is not none %}&middot; cloth quality {{ d.quality }}{% endif %}</span>
 <p><b>Why:</b> {{ why }}</p>
 {% if counts %}<p><b>Defects found:</b> {{ d.findings|length }} ({{ counts }})</p>{% else %}<p><b>Defects found:</b> none</p>{% endif %}</div>
{% if has_img %}<div class="card"><h3>Picture</h3><img src="{{ url_for('report_img', rid=d.id, kind='picture') }}" alt="inspected picture"></div>
{% else %}<div class="card mut">No picture was kept for this inspection (it was logged before reports were added).</div>{% endif %}
{% if has_hm %}<div class="card"><h3>Second-opinion heatmap</h3><img src="{{ url_for('report_img', rid=d.id, kind='heatmap') }}" alt="heatmap"></div>{% endif %}
<div class="card"><h3>Summary of each defect</h3>
{% for x in defects %}<div class="d {{ x.severity }}"><b>#{{ x.n }} {{ x.type }}</b> &middot; {{ x.severity }}<br>
 <b>What:</b> {{ x.size }}<br><b>Where:</b> {{ x.where }}<br><b>Detector confidence:</b> {{ x.confidence }}<br>
 <b>Why this severity:</b> {{ x.why }}<br><b>Check:</b> {{ x.check }}<br><b>Likely cause:</b> {{ x.cause }}<br><b>Route:</b> {{ x.route }}</div>
{% else %}<p>No defect was recorded for this inspection.</p>{% endfor %}</div>
{% if action %}<div class="card"><h3>What to do</h3><div class="kv"><b>Disposition</b><span>{{ action.disposition }}</span>
 <b>Urgency</b><span>{{ action.urgency }}</span></div>{% for n in action.notes %}<p class="mut">{{ n }}</p>{% endfor %}</div>{% endif %}
{% if d.advice and d.advice.source and d.advice.source != 'off' %}<div class="card"><h3>Operator actions <span class="mut">({{ d.advice.source }})</span></h3>
 <ol>{% for x in d.advice.actions %}<li>{{ x }}</li>{% endfor %}</ol>{% if d.advice.risk %}<p><b>Risk:</b> {{ d.advice.risk }}</p>{% endif %}
 {% if d.advice.summary %}<p class="mut">{{ d.advice.summary }}</p>{% endif %}</div>{% endif %}
<div class="card"><h3>Measurement and timing</h3><div class="kv">
 <b>Scale</b><span>{{ ppm }} px/mm{% if ex.calibration_applied is defined and not ex.calibration_applied %} (uploaded photo, not straightened: sizes are estimates){% endif %}</span>
 <b>Camera frame</b><span>{% if ex.input_size %}{{ ex.input_size[0] }} &times; {{ ex.input_size[1] }} px{% else %}-{% endif %}</span>
 <b>Area inspected</b><span>{% if ex.crop %}{{ ex.crop.size[0] }} &times; {{ ex.crop.size[1] }} px at x {{ ex.crop.px[0] }}, y {{ ex.crop.px[1] }}{% else %}whole photo{% endif %}</span>
 <b>Detection</b><span>{{ d.timings.get('detect', '-') }} ms</span><b>Advice</b><span>{{ d.timings.get('llm', '-') }} ms</span>
 <b>Total</b><span>{{ d.timings.get('total', d.timings.get('to_verdict', '-')) }} ms</span></div></div>
{% if d.label %}<div class="card"><h3>Operator confirmation</h3><div class="kv"><b>Result</b><span>{{ d.label }}</span>
 {% if d.corrected %}<b>Corrected verdict</b><span>{{ d.corrected }}</span>{% endif %}{% if d.reason %}<b>Reason</b><span>{{ d.reason }}</span>{% endif %}
 {% if d.note %}<b>Note</b><span>{{ d.note }}</span>{% endif %}</div></div>{% endif %}
</body></html>"""


@app.post("/label")
def label():
    rec = STATE["rec"]
    kind = request.form.get("kind")
    if rec and kind in ("correct", "wrong", "missed"):
        corrected = request.form.get("corrected")
        fpipe.operator_label(rec["id"], kind, request.form.get("reason"), request.form.get("note"),
                             corrected if corrected in VERDICTS else None)
        STATE["label"] = {"kind": kind, "corrected": corrected if corrected in VERDICTS else None}
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



@app.post("/fabric")
def set_fabric():
    k = request.form.get("fabric")
    if k in fabric_choices():
        STATE["fabric"] = k
    return redirect(url_for("index"))


@app.post("/live/start")
def live_start():
    LIVE.start()
    if LIVE.running:
        LIVE.wait_frame()
    return redirect(url_for("index"))


@app.post("/live/stop")
def live_stop():
    LIVE.stop()
    return redirect(url_for("index"))


@app.get("/live.mjpg")
def live_stream():
    resp = Response(LIVE.mjpeg(), mimetype="multipart/x-mixed-replace; boundary=frame")
    resp.headers["Cache-Control"] = "no-store"
    return resp


# ------------------------------------------------------------------ pattern check page

def _straighten(img):
    """Photos from the camera are straightened with the saved calibration. A photo already at the straightened
    size (for example from data/photos_corrected) is left alone."""
    c = fpipe.FAB["fabrics"][fab()]
    ws = tuple(c.get("warp_size") or ())
    if c.get("homography") and (img.shape[1], img.shape[0]) != ws:
        return fcal.apply(img, c)
    return img


def _decode(f):
    img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("%s is not a readable image." % (f.filename or "That file"))
    return img


def _fit(img, width=1100):
    h, w = img.shape[:2]
    return img if w <= width else cv2.resize(img, (width, int(h * width / w)))


def _png(img):
    return cv2.imencode(".png", img)[1].tobytes()


def _draw_gaps(img, gaps):
    out = img.copy()
    r = max(30, int(0.04 * img.shape[1]))
    th = max(2, img.shape[1] // 500)
    for n, (x, y) in enumerate(gaps, 1):
        cv2.circle(out, (int(x), int(y)), r, (0, 0, 255), th)
        cv2.putText(out, "#%d" % n, (int(x) - r, int(y) - r - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    max(0.8, img.shape[1] / 1200.0), (0, 0, 255), 2, cv2.LINE_AA)
    return out


def _draw_compare(cmp):
    ref, test = cmp["ref"].copy(), cmp["warped"].copy()
    th = max(2, ref.shape[1] // 500)
    for n, (x, y, w, h, iou) in enumerate(cmp["diffs"], 1):
        for im in (ref, test):
            cv2.rectangle(im, (x - 6, y - 6), (x + w + 6, y + h + 6), (0, 0, 255), th)
            cv2.putText(im, "#%d" % n, (x, max(20, y - 12)), cv2.FONT_HERSHEY_SIMPLEX, max(0.8, ref.shape[1] / 1200.0),
                        (0, 0, 255), 2, cv2.LINE_AA)
    return np.hstack([_fit(ref, 720), _fit(test, 720)])


PATTERN_PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Pattern check</title>
__STYLE__
<style>.thumbs{display:flex;flex-wrap:wrap;gap:10px}.thumbs div{width:170px}.thumbs img{height:110px;object-fit:cover}
 .pair{display:grid;grid-template-columns:1fr 1fr;gap:6px}</style></head><body>
<header><div><h1>Pattern check &middot; {{ fabric }} &middot; offline</h1>
 <div class="dev"><span>Missing-motif check and comparison with reference photos. Hints for the operator: the main inspection decides the verdict.</span></div></div>
 <a class="btn alt" href="{{ url_for('index') }}" style="text-decoration:none">&larr; Back to inspection</a></header>
{% if error %}<div class="err">{{ error }}</div>{% endif %}
<main>
 <div>
  <div class="card"><h3>1. Reference photos <span class="tag">{{ refs|length }}</span></h3>
   <p class="mut">Add photos of <b>good</b> cloth: 3 to 10 clear photos covering different parts of the cloth, taken with the same
    camera and height. They set the expected motif spacing and let the check compare a test photo with the same part of the cloth.</p>
   <form method="post" action="{{ url_for('pattern_refs') }}" enctype="multipart/form-data" class="row" style="display:flex;gap:8px;flex-wrap:wrap">
    <input type="file" name="refs" accept="image/*" multiple>
    <button name="action" value="add">Add reference photos</button>
    {% if live.running %}<button name="action" value="add_live" class="alt">Add current camera frame</button>{% endif %}</form>
   {% if prof %}<p class="mut">Learned from {{ prof.n_refs }} photo(s): about {{ prof.motifs|round|int }} motifs per photo, motif spacing {{ prof.spacing|round|int }} px.</p>
   {% elif refs %}<p class="mut">No regular motif grid found in the reference photos yet.</p>{% endif %}
   <div class="thumbs">{% for n in refs %}<div><img src="{{ url_for('pattern_ref_img', name=n) }}" alt="{{ n }}">
    <form method="post" action="{{ url_for('pattern_refs') }}"><input type="hidden" name="name" value="{{ n }}">
     <span class="mut">{{ n }}</span> <button name="action" value="delete" class="alt" style="padding:3px 10px;font-size:12px">Remove</button></form></div>{% endfor %}</div></div>
  <div class="card"><h3>2. Photo to check</h3>
   <form method="post" action="{{ url_for('pattern_check') }}" enctype="multipart/form-data" style="display:grid;gap:8px">
    <input type="file" name="photo" accept="image/*">
    <label class="mut"><input type="checkbox" name="compare" value="1" {{ 'checked' if refs else 'disabled' }}> Also compare with the reference photos</label>
    <div class="row" style="display:flex;gap:8px;flex-wrap:wrap"><button name="source" value="upload">Check photo</button>
     {% if live.running %}<button name="source" value="live" class="alt">Check current camera frame</button>{% else %}
     <span class="mut">Start the live view on the inspection page to check a camera frame.</span>{% endif %}</div></form></div>
 </div>
 <div>
 {% if res %}
  <div class="card"><div class="banner {{ 'REVIEW' if res.review else 'PASS' }}" style="font-size:22px;border-radius:6px">
   {{ 'REVIEW: look at the marked spots' if res.review else 'Nothing found by these checks' }}</div>
   <p class="mut">{{ res.ms }} ms &middot; hint only, not a verdict. {{ res.note }}</p>
   {% for w in res.warnings %}<div class="warn">{{ w }}</div>{% endfor %}</div>
  <div class="card"><h3>Missing-motif check</h3>
   <img src="{{ url_for('pattern_img', name='gaps') }}?t={{ res.t }}" alt="Missing motif check">
   <p>{{ res.n_motifs }} motifs found.
    {% if not res.regular %}No regular motif grid in this photo, so this check could not run.
    {% elif res.gaps %}<b>{{ res.gaps|length }} place(s) where a motif seems to be missing</b> (circled).
    {% else %}No missing motif found.{% endif %}</p></div>
  {% if res.compare %}<div class="card"><h3>Comparison with reference photo</h3>
   {% if res.compare.matched %}
    <img src="{{ url_for('pattern_img', name='compare') }}?t={{ res.t }}" alt="Reference and test">
    <p class="mut">Left: reference. Right: your photo lined up on it. Red boxes: motifs whose shape differs.</p>
    <p>Matched reference #{{ res.compare.index + 1 }} ({{ res.compare.inliers }} matching points). {{ res.compare.checked }} motifs compared:
     {% if res.compare.diffs %}<b>{{ res.compare.diffs }} differ</b>.{% else %}none differ.{% endif %}</p>
   {% else %}<p>This photo could not be lined up with any reference, so no comparison was made. Add a reference photo of this part of the cloth.</p>{% endif %}
  </div>{% endif %}
 {% else %}<div class="card">Add reference photos, then upload a photo to check.</div>{% endif %}
 </div>
</main></body></html>"""
PATTERN_PAGE = PATTERN_PAGE.replace("__STYLE__", PAGE[PAGE.index("<style>"):PAGE.index("</style>") + 8])


@app.get("/pattern")
def pattern_page():
    c = fpipe.FAB["fabrics"][fab()]
    refs = [p.name for p in fpat.list_refs(fab())]
    return render_template_string(PATTERN_PAGE, fabric=fab(), refs=refs, live=LIVE, res=PSTATE["result"],
                                  prof=fpat.load_profile(fab(), c["px_per_mm"]) if refs else None,
                                  error=PSTATE.pop("error", None))


@app.post("/pattern/refs")
def pattern_refs():
    act = request.form.get("action")
    try:
        if act == "add":
            n = 0
            for f in request.files.getlist("refs"):
                if f and f.filename:
                    fpat.add_ref(fab(), _straighten(_decode(f)))
                    n += 1
            if not n:
                raise ValueError("Choose one or more photos first.")
        elif act == "add_live":
            img = LIVE.snapshot()
            if img is None:
                raise ValueError("Start the live view first.")
            fpat.add_ref(fab(), _straighten(img))
        elif act == "delete":
            fpat.delete_ref(fab(), request.form.get("name", ""))
    except Exception as e:
        PSTATE["error"] = str(e)
    return redirect(url_for("pattern_page"))


@app.get("/pattern/ref/<name>")
def pattern_ref_img(name):
    p = fpat.ref_dir(fab()) / Path(name).name
    if not p.exists():
        return ("", 404)
    img = cv2.imread(str(p))
    return Response(cv2.imencode(".jpg", _fit(img, 340))[1].tobytes(), mimetype="image/jpeg")


@app.post("/pattern/check")
def pattern_check():
    try:
        if request.form.get("source") == "live":
            img = LIVE.snapshot()
            if img is None:
                raise ValueError("Start the live view first.")
        else:
            f = request.files.get("photo")
            if not f or not f.filename:
                raise ValueError("Choose a photo first.")
            img = _decode(f)
        t0 = time.perf_counter()
        img = _straighten(img)
        ppm = fpipe.FAB["fabrics"][fab()]["px_per_mm"]
        prof = fpat.load_profile(fab(), ppm)
        g = fpat.find_gaps(img, ppm, prof)
        warnings = []
        if g["spacing_ratio"] is not None and not 0.85 <= g["spacing_ratio"] <= 1.15:
            warnings.append("The motifs are %.0f%% of the size seen in the reference photos. Check the camera height or "
                            "that this is the same cloth." % (100 * g["spacing_ratio"]))
        res = {"n_motifs": g["n_motifs"], "regular": g["regular"], "gaps": g["gaps"], "warnings": warnings,
               "compare": None, "t": int(time.time()), "note": ""}
        PSTATE["gaps_png"] = _png(_fit(_draw_gaps(img, g["gaps"])))
        refs = [cv2.imread(str(p)) for p in fpat.list_refs(fab())]
        if request.form.get("compare") and refs:
            cmp = fpat.compare_to_reference(img, refs, ppm)
            res["compare"] = {"matched": cmp["matched"], "inliers": cmp["inliers"], "checked": cmp["checked"],
                              "diffs": len(cmp["diffs"]), "index": cmp["ref_index"] or 0}
            if cmp["matched"]:
                PSTATE["compare_png"] = _png(_draw_compare(cmp))
        res["review"] = bool(g["gaps"]) or bool(res["compare"] and res["compare"]["diffs"])
        res["note"] = ("Measured on our pattern cloth with made-up defects; see README." if res["review"] else
                       "A clean result does not prove the cloth is perfect: wrong-type or flipped motifs are not checked.")
        res["ms"] = int((time.perf_counter() - t0) * 1000)
        PSTATE["result"] = res
    except Exception as e:
        PSTATE["error"] = str(e)
    return redirect(url_for("pattern_page"))


@app.get("/pattern/img/<name>")
def pattern_img(name):
    data = PSTATE.get("gaps_png" if name == "gaps" else "compare_png")
    if not data:
        return ("", 404)
    resp = Response(data, mimetype="image/png")
    resp.headers["Cache-Control"] = "no-store"
    return resp


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
