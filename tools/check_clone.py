"""Pre-demo check for a fresh clone: are the files the fabric entry needs really there?

Run from the repo root:   python tools/check_clone.py
Exit code 0 = ready, 1 = something is missing. It never downloads or changes anything.
Checks the blue-denim entry (detector, anomaly files, Llama files) and flags Git LFS pointer
stubs, which look like model files but are only a few hundred bytes.
"""
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
MIN_MODEL_BYTES = 100_000          # an LFS pointer stub is about 130 bytes


def check_file(rel, min_bytes=MIN_MODEL_BYTES):
    p = ROOT / rel
    if not p.exists():
        return False, "missing"
    if p.stat().st_size < min_bytes:
        return False, "only %d bytes (Git LFS pointer? run: git lfs pull)" % p.stat().st_size
    return True, "%.1f MB" % (p.stat().st_size / 1e6)


def main():
    fab = yaml.safe_load((ROOT / "kb" / "fabric.yaml").read_text(encoding="utf-8"))
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    name = fab.get("fabric", "blue")
    f = fab["fabrics"][name]
    llm = cfg.get("llm", {})
    required = [("detector", f.get("yolo_model"), MIN_MODEL_BYTES)]
    a = f.get("anomaly") or {}
    if a.get("enabled"):
        required += [("anomaly backbone", a.get("backbone_onnx"), MIN_MODEL_BYTES),
                     ("anomaly bank", a.get("bank"), MIN_MODEL_BYTES)]
    if llm.get("enabled", True):
        required += [("Llama .pte", llm.get("pte"), 100_000_000),
                     ("Llama tokenizer", llm.get("tokenizer"), 100_000)]
    bad = 0
    print("Fabric: %s | anomaly: %s | Llama: %s" % (name, "on" if a.get("enabled") else "off",
                                                     "on" if llm.get("enabled", True) else "off"))
    for label, rel, mn in required:
        if not rel:
            print("  [FAIL] %-17s not set in the config" % label); bad += 1; continue
        ok, why = check_file(rel, mn)
        print("  [%s] %-17s %s  (%s)" % ("ok" if ok else "FAIL", label, rel, why))
        bad += 0 if ok else 1
    ppm = f.get("px_per_mm")
    print("  [%s] calibration      px_per_mm=%s calibrated=%s" % (
        "ok" if f.get("calibrated") else "WARN", ppm, f.get("calibrated")))
    try:
        import fabric_pipeline  # noqa: F401
        print("  [ok] fabric_pipeline imports")
    except Exception as e:
        print("  [FAIL] fabric_pipeline import: %s" % e); bad += 1
    runner = Path.home() / "executorch" / "cmake-out" / "examples" / "models" / "llama" / "llama_main"
    if llm.get("enabled", True) and not runner.exists():
        print("  [WARN] Llama runner not found at %s (expected on the Pi only)" % runner)
    print("READY" if bad == 0 else "NOT READY: %d problem(s)" % bad)
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    sys.exit(main())
