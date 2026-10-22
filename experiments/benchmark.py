"""E2 benchmark: run the full pipeline N times, record per-stage latency,
total latency, peak RSS memory and CPU temperature (Pi) to CSV.

Usage:  python -m experiments.benchmark --runs 20 --tag laptop_cpu
"""
import argparse
import csv
import statistics
import time
from datetime import datetime
from pathlib import Path

import psutil

import pipeline


def cpu_temp_c():
    try:
        return round(int(Path("/sys/class/thermal/thermal_zone0/temp").read_text()) / 1000, 1)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--tag", default="laptop_cpu")
    a = ap.parse_args()

    proc = psutil.Process()
    for _ in range(a.warmup):
        pipeline.run_once(verbose=False)

    out = Path("results") / f"bench_{a.tag}_{datetime.now():%Y%m%d_%H%M%S}.csv"
    out.parent.mkdir(exist_ok=True)
    rows, peak_rss = [], 0
    for i in range(a.runs):
        r = pipeline.run_once(verbose=False)
        rss = proc.memory_info().rss / 1e6
        peak_rss = max(peak_rss, rss)
        rows.append({"run": i, **r["timings"], "rss_mb": round(rss, 1),
                     "cpu_temp_c": cpu_temp_c(), "verdict": r["advice"]["verdict"],
                     "advice_source": r["advice"].get("source")})
        time.sleep(0.05)

    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    totals = [r["total"] for r in rows]
    print(f"runs={a.runs} tag={a.tag}")
    print(f"total ms: median={statistics.median(totals):.1f} "
          f"p95={sorted(totals)[int(0.95 * (len(totals) - 1))]:.1f} max={max(totals):.1f}")
    for stage in [k for k in rows[0] if k not in ("run", "total", "rss_mb", "cpu_temp_c",
                                                   "verdict", "advice_source")]:
        print(f"  {stage:10} median {statistics.median(r[stage] for r in rows):8.2f} ms")
    print(f"peak RSS: {peak_rss:.0f} MB | saved {out}")


if __name__ == "__main__":
    main()
