"""Operator UI (stage 7): local web page, works fully offline.

Run from the repo root:  python -m ui.app   then open http://127.0.0.1:5000
On the Pi, open http://<pi-ip>:5000 from any device on the same network.
"""
import json
import sqlite3
from pathlib import Path

from flask import Flask, redirect, render_template_string, request, send_file, url_for

import pipeline

app = Flask(__name__)
LAST = {}

PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PCB Inspection Station</title>
<style>
 body{font-family:Arial,sans-serif;margin:0;background:#111;color:#eee}
 header{display:flex;justify-content:space-between;align-items:center;padding:12px 20px;background:#1c1c1c}
 h1{font-size:18px;margin:0}
 select{font-size:15px;padding:9px;border-radius:6px;background:#2a2a2a;color:#eee;border:1px solid #444;margin-right:8px}
 .err{background:#5a1d1d}
 button{font-size:16px;padding:10px 22px;border:0;border-radius:6px;background:#2d7ff9;color:#fff;cursor:pointer}
 main{display:grid;grid-template-columns:minmax(0,3fr) minmax(0,2fr);gap:16px;padding:16px}
 @media(max-width:900px){main{grid-template-columns:1fr}}
 .card{background:#1c1c1c;border-radius:8px;padding:14px}
 img{width:100%;border-radius:6px}
 .badge{display:inline-block;padding:6px 16px;border-radius:6px;font-weight:bold;font-size:22px}
 .PASS{background:#1f8b3a}.REWORK{background:#d98200}.REJECT{background:#c62828}
 .CRITICAL{color:#ff5252}.MAJOR{color:#ffa726}.MINOR{color:#66bb6a}
 table{width:100%;border-collapse:collapse;font-size:14px}
 td,th{text-align:left;padding:6px;border-bottom:1px solid #333}
 li{margin-bottom:6px}.muted{color:#999;font-size:13px}
</style></head><body>
<header><h1>PCB Inspection Station &middot; offline</h1>
<form method="post" action="{{ url_for('inspect') }}">
 <select name="board" aria-label="Board">{% for p in boards %}
  <option value="{{ p.id }}" {% if p.id == selected %}selected{% endif %} {% if not p.ready %}disabled{% endif %}>
   {{ p.name }}{% if not p.ready %} (not ready){% endif %}</option>{% endfor %}
 </select><button>Inspect board</button></form></header>
{% if error %}<div style="padding:16px 16px 0"><div class="card err">{{ error }}</div></div>{% endif %}
{% set notready = boards|rejectattr('ready')|list %}{% if notready %}
<div style="padding:16px 16px 0"><div class="card muted">{% for p in notready %}
 <div><b>{{ p.name }}</b> not ready: {{ p.problems()|join('; ') }}</div>{% endfor %}</div></div>{% endif %}
{% if r %}
<main>
 <div class="card"><img src="{{ url_for('image') }}?t={{ r.timings.total }}" alt="Inspected board"></div>
 <div>
  <div class="card"><p class="muted">Board: {{ r.board_name }}</p><span class="badge {{ r.advice.verdict }}">{{ r.advice.verdict }}</span>
   <p>{{ r.advice.summary }}</p><p class="muted">Risk: {{ r.advice.risk }}</p></div>
  <div class="card"><h3>Defects</h3><table>
   <tr><th>Ref</th><th>Defect</th><th>Pins / nets</th><th>Severity</th><th>Rule</th></tr>
   {% for d in r.defects %}<tr><td>{{ d.refdes }}</td><td>{{ d.type }}</td>
    <td>{{ d.pins }} {{ d.nets|join('/') }}</td><td class="{{ d.severity }}">{{ d.severity }}</td>
    <td>{{ d.rule_id }}</td></tr>{% endfor %}</table></div>
  <div class="card"><h3>Operator actions</h3><ol>{% for a in r.advice.actions %}<li>{{ a }}</li>{% endfor %}</ol>
   <p class="muted">Advice source: {{ r.advice.source }} &middot; total {{ r.timings.total }} ms</p></div>
 </div>
</main>
{% else %}<main><div class="card">Place a board in the fixture and press <b>Inspect board</b>.</div></main>{% endif %}
<div style="padding:0 16px 16px"><div class="card"><h3>Recent inspections</h3><table>
 <tr><th>Time</th><th>Board</th><th>Verdict</th><th>Defects</th><th>Total ms</th></tr>
 {% for h in history %}<tr><td>{{ h.ts }}</td><td>{{ h.board }}</td><td>{{ h.verdict }}</td><td>{{ h.n }}</td><td>{{ h.ms }}</td></tr>{% endfor %}
</table></div></div></body></html>"""


def history(limit=10):
    db = Path(pipeline.CFG["paths"]["db"])
    if not db.exists():
        return []
    con = sqlite3.connect(db)
    cols = [r[1] for r in con.execute("PRAGMA table_info(inspections)")]
    board = "board" if "board" in cols else "''"
    rows = con.execute(f"SELECT ts, {board}, verdict, defects, timings FROM inspections "
                       "ORDER BY rowid DESC LIMIT ?", (limit,)).fetchall()
    con.close()
    return [{"ts": ts, "board": b or "-", "verdict": v, "n": len(json.loads(d)),
             "ms": json.loads(t).get("total", "-")} for ts, b, v, d, t in rows]


@app.get("/")
def index():
    selected = LAST.get("board") or pipeline.CFG["board"]
    return render_template_string(PAGE, r=LAST.get("r"), history=history(),
                                  boards=list(pipeline.PROFILES.values()),
                                  selected=selected, error=LAST.pop("error", None))


@app.post("/inspect")
def inspect():
    board = request.form.get("board") or pipeline.CFG["board"]
    LAST["board"] = board
    try:
        LAST["r"] = pipeline.run_once(board, verbose=False)
    except (pipeline.ProfileError, RuntimeError) as e:
        LAST["error"] = str(e)
    return redirect(url_for("index"))


@app.get("/image")
def image():
    return send_file(Path(LAST["r"]["image"]).resolve(), mimetype="image/png")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
