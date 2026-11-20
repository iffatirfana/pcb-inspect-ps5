"""Timing, thread and heat measurements for the Raspberry Pi 5 (also runs on a laptop, without the Pi-only readings).

Run from the repository root, with the virtual environment active:

    python tools/pi_benchmark.py latency  --n 30            # p50 / p95 / p99 for every stage
    python tools/pi_benchmark.py threads  --threads 1,2,3,4 # best ONNX Runtime thread count
    python tools/pi_benchmark.py parallel --n 10            # detector and anomaly: one after the other vs together
    python tools/pi_benchmark.py soak     --minutes 30      # long run: temperature, clock, throttling, memory
    python tools/pi_benchmark.py all                        # all of the above (about 45 minutes)

Photos come from --photos (default data/photos_corrected, ruler photos skipped). They are treated as already
straightened; add --camera-style to straighten them first. Results go to results/bench/ (CSV files and
summary.md). The inspections run against a throw-away database, so the dashboard log is not touched.
The Llama stage is left out unless --llm is given.
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
OUT = Path("results/bench")


def photos(folder, limit=None):
    import cv2
    files = sorted(p for p in Path(folder).glob("*.png") if not p.name.startswith("ruler"))
    if not files:
        sys.exit("No photos found in %s" % folder)
    return [cv2.imread(str(p)) for p in files[:limit]]


def pct(values, q):
    return round(float(np.percentile(values, q)), 1) if len(values) else None


def stats(values):
    return {"n": len(values), "mean": round(float(np.mean(values)), 1) if len(values) else None,
            "p50": pct(values, 50), "p95": pct(values, 95), "p99": pct(values, 99),
            "max": round(float(np.max(values)), 1) if len(values) else None}


def setup_pipeline(fabric):
    import fabric_pipeline as fpipe
    tmp = Path(tempfile.mkdtemp(prefix="bench_"))
    fpipe.CFG["paths"]["db"] = str(tmp / "bench.sqlite")        # keep the dashboard log clean
    fpipe.REPORT_DIR = tmp / "reports"
    fpipe.FAB["fabric"] = fabric
    return fpipe


def run_inspections(fpipe, imgs, fabric, n, calibrate, llm, warmup=2):
    """Inspect n photos (cycling through imgs). Returns a list of timing dicts in ms, warm-up runs left out."""
    rows = []
    for i in range(n + warmup):
        t = time.perf_counter()
        rec = fpipe.inspect(fabric, imgs[i % len(imgs)], calibrate=calibrate)
        if llm and rec["verdict"] not in ("PASS", fpipe.INVALID):
            fpipe.advise(rec)
        wall = (time.perf_counter() - t) * 1000
        if i >= warmup:
            row = dict(rec["timings"])
            row["wall"] = round(wall, 2)
            row["verdict"] = rec["verdict"]
            rows.append(row)
    return rows


def summarise(rows):
    keys = ["detect", "anomaly", "pattern", "to_verdict", "llm", "total", "wall"]
    return {k: stats([r[k] for r in rows if k in r]) for k in keys if any(k in r for r in rows)}


def write_csv(name, rows):
    OUT.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = sorted({k for r in rows for k in r})
    with open(OUT / name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def add_summary(title, body):
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "summary.md", "a", encoding="utf-8") as f:
        f.write("\n## %s\n\n_%s_\n\n%s\n" % (title, datetime.now().strftime("%Y-%m-%d %H:%M"), body))


def table(rows, header):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


# ------------------------------------------------------------------ latency
def cmd_latency(a):
    fpipe = setup_pipeline(a.fabric)
    imgs = photos(a.photos)
    rows = run_inspections(fpipe, imgs, a.fabric, a.n, a.camera_style, a.llm)
    s = summarise(rows)
    if a.json:                                  # machine-readable line for the thread sweep
        print("RESULT " + json.dumps({"threads": os.environ.get("FABRIC_ORT_THREADS", "default"), "stats": s}))
        return s
    write_csv("latency.csv", rows)
    body = table([[k, v["n"], v["mean"], v["p50"], v["p95"], v["p99"], v["max"]] for k, v in s.items()],
                 ["Stage (ms)", "n", "mean", "p50", "p95", "p99", "max"])
    verdicts = {}
    for r in rows:
        verdicts[r["verdict"]] = verdicts.get(r["verdict"], 0) + 1
    body += "\n\nVerdicts in this run: %s. Warm-up runs excluded." % verdicts
    add_summary("Latency per stage (%d photos from %s)" % (a.n, a.photos), body)
    print(body)
    return s


# ------------------------------------------------------------------ threads
def cmd_threads(a):
    results = []
    for t in [x.strip() for x in a.threads.split(",")]:
        env = dict(os.environ, FABRIC_ORT_THREADS=t)
        cmd = [sys.executable, __file__, "latency", "--n", str(a.n), "--photos", a.photos, "--fabric", a.fabric,
               "--json"] + (["--camera-style"] if a.camera_style else [])
        p = subprocess.run(cmd, env=env, capture_output=True, text=True)
        line = next((l for l in p.stdout.splitlines() if l.startswith("RESULT ")), None)
        if not line:
            print("threads=%s failed:\n%s" % (t, p.stderr[-500:]))
            continue
        s = json.loads(line[7:])["stats"]
        results.append((t, s))
        print("threads", t, "to_verdict p50", s.get("to_verdict", {}).get("p50"), "ms")
    rows = [[t, s["to_verdict"]["p50"], s["to_verdict"]["p95"], s.get("detect", {}).get("p50", "-"),
             s.get("anomaly", {}).get("p50", "-")] for t, s in results]
    body = table(rows, ["ONNX threads", "photo-to-verdict p50 (ms)", "p95 (ms)", "detector p50 (ms)", "anomaly p50 (ms)"])
    if rows:
        best = min(results, key=lambda r: r[1]["to_verdict"]["p50"])[0]
        body += "\n\nFastest median: %s thread(s)." % best
    add_summary("ONNX Runtime thread count", body)
    print(body)


# ------------------------------------------------------------------ parallel
def cmd_parallel(a):
    from concurrent.futures import ThreadPoolExecutor
    fpipe = setup_pipeline(a.fabric)
    import fabric_profile as fp
    c = fpipe.FAB["fabrics"][a.fabric]
    det, sc = fpipe.get_detector(a.fabric), fpipe.get_anomaly_scorer(a.fabric)
    if sc is None:
        sys.exit("The anomaly second opinion is off or its files are missing for fabric '%s'." % a.fabric)
    img = fp.crop_image(photos(a.photos, 1)[0], c.get("crop"))
    for _ in range(2):                                          # warm-up
        det.detect(img); sc.heatmap(img)
    seq, par = [], []
    with ThreadPoolExecutor(2) as pool:
        for _ in range(a.n):
            t = time.perf_counter(); det.detect(img); sc.heatmap(img)
            seq.append((time.perf_counter() - t) * 1000)
            t = time.perf_counter(); f1, f2 = pool.submit(det.detect, img), pool.submit(sc.heatmap, img)
            f1.result(); f2.result()
            par.append((time.perf_counter() - t) * 1000)
    s, p = stats(seq), stats(par)
    body = table([["one after the other", s["p50"], s["p95"], s["mean"]], ["both at the same time", p["p50"], p["p95"], p["mean"]]],
                 ["Detector + anomaly", "p50 (ms)", "p95 (ms)", "mean (ms)"])
    body += "\n\nThread setting: %s." % os.environ.get("FABRIC_ORT_THREADS", "default (all cores)")
    add_summary("Detector and anomaly: sequential vs parallel", body)
    print(body)


# ------------------------------------------------------------------ soak
def sh(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:
        return ""


def read_cpu():
    with open("/proc/stat") as f:
        return [list(map(int, l.split()[1:])) for l in f if l.startswith("cpu") and l[3].isdigit()]


def core_util(prev, cur):
    out = []
    for a, b in zip(prev, cur):
        total, idle = sum(b) - sum(a), (b[3] + b[4]) - (a[3] + a[4])
        out.append(round(100.0 * (total - idle) / total, 1) if total else 0.0)
    return out


def mem_status():
    rss = hwm = swap = None
    try:
        for l in open("/proc/self/status"):
            if l.startswith("VmRSS"): rss = int(l.split()[1]) / 1024
            if l.startswith("VmHWM"): hwm = int(l.split()[1]) / 1024
        m = {l.split(":")[0]: int(l.split()[1]) for l in open("/proc/meminfo")}
        swap = (m["SwapTotal"] - m["SwapFree"]) / 1024
    except Exception:
        pass
    return rss, hwm, swap


def cmd_soak(a):
    fpipe = setup_pipeline(a.fabric)
    imgs = photos(a.photos)
    end, samples, lat = time.time() + a.minutes * 60, [], []
    stop = threading.Event()

    def sampler():
        prev = read_cpu() if Path("/proc/stat").exists() else []
        while not stop.wait(a.interval):
            cur = read_cpu() if prev else []
            temp = sh(["vcgencmd", "measure_temp"]).replace("temp=", "").replace("'C", "")
            clock = sh(["vcgencmd", "measure_clock", "arm"]).split("=")[-1]
            thr = sh(["vcgencmd", "get_throttled"]).split("=")[-1]
            rss, hwm, swap = mem_status()
            samples.append({"t_s": round(time.time() - start), "temp_c": temp, "arm_clock_hz": clock, "throttled": thr,
                            "rss_mb": round(rss, 1) if rss else "", "peak_rss_mb": round(hwm, 1) if hwm else "",
                            "swap_used_mb": round(swap, 1) if swap is not None else "",
                            "core_util_pct": " ".join(map(str, core_util(prev, cur))) if cur else ""})
            prev = cur or prev

    start = time.time()
    th = threading.Thread(target=sampler, daemon=True)
    th.start()
    i = 0
    while time.time() < end:
        t = time.perf_counter()
        rec = fpipe.inspect(a.fabric, imgs[i % len(imgs)], calibrate=a.camera_style)
        if a.llm and rec["verdict"] not in ("PASS", fpipe.INVALID):
            fpipe.advise(rec)
        lat.append((time.perf_counter() - t) * 1000)
        i += 1
    stop.set(); th.join(timeout=a.interval + 2)
    write_csv("soak_samples.csv", samples)
    temps = [float(s["temp_c"]) for s in samples if s["temp_c"]]
    clocks = [int(s["arm_clock_hz"]) / 1e6 for s in samples if s["arm_clock_hz"].isdigit()]
    thr = sorted({s["throttled"] for s in samples if s["throttled"]})
    swaps = [s["swap_used_mb"] for s in samples if s["swap_used_mb"] != ""]
    first, last = lat[: max(1, len(lat) // 10)], lat[-max(1, len(lat) // 10):]
    rows = [["Duration", "%.1f min" % ((time.time() - start) / 60)], ["Inspections", len(lat)],
            ["Latency p50 / p95 / p99 (ms)", "%s / %s / %s" % (pct(lat, 50), pct(lat, 95), pct(lat, 99))],
            ["Median latency, first vs last tenth of the run (ms)", "%s vs %s" % (pct(first, 50), pct(last, 50))],
            ["Temperature min / max (C)", "%s / %s" % (min(temps), max(temps)) if temps else "n/a (no vcgencmd)"],
            ["ARM clock min / max (MHz)", "%.0f / %.0f" % (min(clocks), max(clocks)) if clocks else "n/a (no vcgencmd)"],
            ["get_throttled values seen", ", ".join(thr) if thr else "n/a (no vcgencmd)"],
            ["Peak memory of this process (MB)", max([s["peak_rss_mb"] for s in samples if s["peak_rss_mb"] != ""] or ["n/a"])],
            ["Peak swap used (MB)", max(swaps) if swaps else "n/a"]]
    body = table(rows, ["Measurement", "Value"]) + "\n\nA throttled value of 0x0 for the whole run means no under-voltage or thermal limiting."
    add_summary("Sustained run (%s min)" % a.minutes, body)
    print(body)


def cmd_all(a):
    a.n = a.n or 30
    cmd_latency(a)
    a.threads, a.n = "1,2,3,4", 10
    cmd_threads(a)
    cmd_parallel(a)
    a.minutes = 30
    cmd_soak(a)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=["latency", "threads", "parallel", "soak", "all"])
    p.add_argument("--photos", default="data/photos_corrected")
    p.add_argument("--fabric", default="blue")
    p.add_argument("--n", type=int, default=30, help="number of inspections")
    p.add_argument("--threads", default="1,2,3,4")
    p.add_argument("--minutes", type=float, default=30)
    p.add_argument("--interval", type=float, default=10, help="seconds between heat and memory readings")
    p.add_argument("--camera-style", action="store_true", help="straighten the photos with the calibration first")
    p.add_argument("--llm", action="store_true", help="include the Llama stage (needs the runner set up)")
    p.add_argument("--json", action="store_true")
    a = p.parse_args()
    {"latency": cmd_latency, "threads": cmd_threads, "parallel": cmd_parallel, "soak": cmd_soak, "all": cmd_all}[a.cmd](a)


if __name__ == "__main__":
    main()
