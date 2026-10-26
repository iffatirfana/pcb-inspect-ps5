"""Operator UI: local web page, works fully offline.

Run from the repo root:  python -m ui.app   then open http://127.0.0.1:5000
On the Pi, open http://<pi-ip>:5000 from any device on the same network.

Flow: Inspect -> verdict + defects shown at once -> Llama advice loads in the
background -> operator marks each finding correct / wrong, or reports a miss.
"""
import json
import sqlite3
import threading
from pathlib import Path

from flask import Flask, jsonify, redirect, render_template_string, request, send_file, url_for

import pipeline

app = Flask(__name__)
LAST = {}
LOCK = threading.Lock()

PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PCB Inspection Station</title>
<style>
 body{font-family:Arial,sans-serif;margin:0;background:#111;color:#eee}
 header{display:flex;flex-wrap:wrap;gap:10px;justify-content:space-between;align-items:center;padding:12px 20px;background:#1c1c1c}
 h1{font-size:18px;margin:0} h3{margin:4px 0 10px}
 select{font-size:15px;padding:9px;border-radius:6px;background:#2a2a2a;color:#eee;border:1px solid #444;margin-right:8px}
 button{font-size:16px;padding:10px 22px;border:0;border-radius:6px;background:#2d7ff9;color:#fff;cursor:pointer}
 button.s{font-size:12px;padding:4px 9px;margin:1px;background:#333}
 button.ok{background:#1f6b33} button.no{background:#7a2323}
 main{display:grid;grid-template-columns:minmax(0,3fr) minmax(0,2fr);gap:16px;padding:16px}
 @media(max-width:900px){main{grid-template-columns:1fr}}
 .card{background:#1c1c1c;border-radius:8px;padding:14px;margin-bottom:14px}
 img{width:100%;border-radius:6px}
 .badge{display:inline-block;padding:6px 16px;border-radius:6px;font-weight:bold;font-size:22px}
 .PASS{background:#1f8b3a}.REWORK{background:#d98200}.REJECT{background:#c62828}
 .REVIEW{background:#7b4fa0}.INVALID_CAPTURE{background:#555}
 .CRITICAL{color:#ff5252}.MAJOR{color:#ffa726}.MINOR{color:#66bb6a}
 table{width:100%;border-collapse:collapse;font-size:14px}
 td,th{text-align:left;padding:6px;border-bottom:1px solid #333;vertical-align:top}
 li{margin-bottom:6px}.muted{color:#999;font-size:13px}.err{background:#5a1d1d}
 input[type=text]{background:#2a2a2a;color:#eee;border:1px solid #444;border-radius:6px;padding:7px;width:60%}
</style></head><body>
<header><h1>PCB Inspection Station &middot; offline</h1>
<form method="post" action="{{ url_for('inspect') }}">
 <select name="board" aria-label="Board">{% for p in boards %}
  <option value="{{ p.id }}" {% if p.id == selected %}selected{% endif %} {% if not p.ready %}disabled{% endif %}>
   {{ p.name }} ({{ p.board_type }}){% if not p.ready %} - not ready{% endif %}</option>{% endfor %}
 </select><button>Inspect board</button></form></header>
{% if error %}<div style="padding:16px 16px 0"><div class="card err">{{ error }}</div></div>{% endif %}
{% set notready = boards|rejectattr('ready')|list %}{% if notready %}
<div style="padding:16px 16px 0"><div class="card muted">{% for p in notready %}
 <div><b>{{ p.name }}</b> not ready: {{ p.problems()|join('; ') }}</div>{% endfor %}</div></div>{% endif %}
{% if r %}
<main>
 <div>{% if r.image %}<div class="card"><img src="{{ url_for('image') }}?id={{ r.id }}" alt="Inspected board"></div>{% endif %}</div>
 <div>
  <div class="card"><p class="muted">Board: {{ r.board_name }} &middot; {{ r.board_type }} &middot; inspector: {{ r.inspector }} &middot; #{{ r.id }}</p>
   <span class="badge {{ r.verdict }}">{{ r.verdict|replace('_',' ') }}</span>
   {% if r.invalid_reasons %}<ul>{% for x in r.invalid_reasons %}<li>{{ x }}</li>{% endfor %}</ul>
   <p class="muted">No verdict on this photo - fix the problem and inspect again.</p>{% endif %}
   <p class="muted">Verdict in {{ r.timings.to_verdict }} ms</p></div>
  {% if not r.invalid_reasons %}
  <div class="card"><h3>Defects</h3>{% if r.defects %}<table>
   <tr><th>Ref</th><th>Defect</th><th>Pins / nets</th><th>Severity</th><th>Verdict</th><th>Check</th></tr>
   {% for d in r.defects %}<tr><td>{{ d.refdes }}</td>
    <td>{{ d.type|replace('_',' ') }}<br><span class="muted">{{ d.source }} {{ '%.2f'|format(d.conf) if d.conf is not none }}</span></td>
    <td>{{ d.pins }} {{ d.nets|join('/') }}</td><td class="{{ d.severity }}">{{ d.severity }}</td>
    <td>{{ d.verdict }}<br><span class="muted">{{ d.rule_id }}</span></td>
    <td>{% set v = marks.get(loop.index0) %}{% if v %}<span class="muted">marked {{ v }}</span>{% else %}
     <form method="post" action="{{ url_for('verify') }}" style="display:inline">
      <input type="hidden" name="id" value="{{ r.id }}"><input type="hidden" name="idx" value="{{ loop.index0 }}">
      <button class="s ok" name="label" value="correct" title="Real defect">&#10003;</button>
      <button class="s no" name="label" value="wrong" title="False alarm">&#10007;</button></form>{% endif %}</td></tr>{% endfor %}
   </table>{% else %}<p>No defects found.</p>{% endif %}
   <form method="post" action="{{ url_for('verify') }}" style="margin-top:10px">
    <input type="hidden" name="id" value="{{ r.id }}"><input type="hidden" name="idx" value="-1">
    <input type="text" name="note" placeholder="Missed defect? e.g. bridge on U2 pins 3-4" required>
    <button class="s" name="label" value="missed">Report miss</button></form></div>
  {% endif %}
  <div class="card" id="advice"><h3>Operator actions</h3><p class="muted">Preparing advice&hellip;</p></div>
 </div>
</main>
<script>
async function poll(){
  const r = await fetch("{{ url_for('advice') }}?id={{ r.id }}"); const j = await r.json();
  if(!j.ready){ setTimeout(poll, 800); return; }
  const a = j.advice, el = document.getElementById('advice'), esc = s => String(s).replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
  el.innerHTML = '<h3>Operator actions</h3><p>' + esc(a.summary) + '</p><p class="muted">Risk: ' + esc(a.risk) + '</p><ol>' +
    a.actions.map(x => '<li>' + esc(x) + '</li>').join('') + '</ol><p class="muted">Advice source: ' + esc(a.source) +
    ' &middot; ' + j.llm_ms + ' ms</p>';
}
poll();
</script>
{% else %}<main><div class="card">Place a board in the fixture, choose it above and press <b>Inspect board</b>.</div></main>{% endif %}
<div style="padding:0 16px 16px"><div class="card"><h3>Recent inspections</h3><table>
 <tr><th>#</th><th>Time</th><th>Board</th><th>Verdict</th><th>Defects</th><th>To verdict ms</th></tr>
 {% for h in history %}<tr><td>{{ h.id }}</td><td>{{ h.ts }}</td><td>{{ h.board }}</td><td>{{ h.verdict }}</td><td>{{ h.n }}</td><td>{{ h.ms }}</td></tr>{% endfor %}
</table></div></div></body></html>"""


def _con():
    db = Path(pipeline.CFG["paths"]["db"])
    return sqlite3.connect(db) if db.exists() else None


def history(limit=10):
    con = _con()
    if con is None:
        return []
    cols = {r[1] for r in con.execute("PRAGMA table_info(inspections)")}
    board = "board" if "board" in cols else "''"
    rows = con.execute(f"SELECT rowid, ts, {board}, verdict, defects, timings FROM inspections "
                       "ORDER BY rowid DESC LIMIT ?", (limit,)).fetchall()
    con.close()
    out = []
    for rid, ts, b, v, d, t in rows:
        t = json.loads(t or "{}")
        out.append({"id": rid, "ts": ts, "board": b or "-", "verdict": v, "n": len(json.loads(d or "[]")),
                    "ms": t.get("to_verdict", t.get("total", "-"))})
    return out


def marks(inspection_id):
    con = _con()
    if con is None or inspection_id is None:
        return {}
    try:
        rows = con.execute("SELECT defect_index, label FROM verifications WHERE inspection_id=?",
                           (inspection_id,)).fetchall()
    except sqlite3.OperationalError:
        rows = []
    con.close()
    return {i: lab for i, lab in rows if i >= 0}


def _advise_async(rec):
    try:
        pipeline.advise(rec)
    except Exception as e:                      # never leave the page waiting forever
        rec["advice"] = {"summary": f"Advice unavailable ({e.__class__.__name__}).", "risk": "-",
                         "actions": ["Follow the verdict and the SOP for each defect."],
                         "verdict": rec["verdict"], "source": "error"}
        rec["timings"]["llm"] = 0


@app.get("/")
def index():
    r = LAST.get("r")
    return render_template_string(PAGE, r=r, history=history(),
                                  boards=list(pipeline.PROFILES.values()),
                                  selected=LAST.get("board") or pipeline.CFG["board"],
                                  error=LAST.pop("error", None),
                                  marks=marks(r["id"]) if r else {})


@app.post("/inspect")
def inspect():
    board = request.form.get("board") or pipeline.CFG["board"]
    LAST["board"] = board
    try:
        with LOCK:                              # one inspection at a time (one camera)
            rec = pipeline.inspect(board)
        LAST["r"] = rec
        threading.Thread(target=_advise_async, args=(rec,), daemon=True).start()
    except (pipeline.ProfileError, pipeline.InspectorError, RuntimeError) as e:
        LAST["error"] = str(e)
    return redirect(url_for("index"))


@app.get("/advice")
def advice():
    r = LAST.get("r")
    if not r or str(r["id"]) != request.args.get("id") or r.get("advice") is None:
        return jsonify(ready=False)
    return jsonify(ready=True, advice=r["advice"], llm_ms=r["timings"].get("llm"))


@app.post("/verify")
def verify():
    pipeline.verify(int(request.form["id"]), int(request.form["idx"]),
                    request.form["label"], request.form.get("note", ""))
    return redirect(url_for("index"))


@app.get("/image")
def image():
    return send_file(Path(LAST["r"]["image"]).resolve(), mimetype="image/png")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True)
