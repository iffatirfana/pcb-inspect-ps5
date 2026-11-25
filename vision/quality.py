"""Image-quality gate: refuse to give a verdict on a bad photo.

One cheap check:
  check_capture(img)      sharpness, exposure, clipping
Any failure -> verdict INVALID_CAPTURE (operator re-takes the photo).
Limits live in config.yaml -> quality.
"""
import cv2
import numpy as np

DEFAULTS = {"min_sharpness": 40.0,        # variance of Laplacian at 1024 px width
            "min_brightness": 35, "max_brightness": 225,
            "max_clipped": 0.10}          # fraction of pixels at 0-4 or 251-255


def _limits(cfg):
    return {**DEFAULTS, **(cfg or {})}


def capture_metrics(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    if g.shape[1] > 1024:
        g = cv2.resize(g, (1024, round(g.shape[0] * 1024 / g.shape[1])), interpolation=cv2.INTER_AREA)
    return {"sharpness": round(float(cv2.Laplacian(g, cv2.CV_64F).var()), 1),
            "brightness": round(float(g.mean()), 1),
            "clipped": round(float(((g < 5) | (g > 250)).mean()), 3)}


def check_capture(img, cfg=None):
    """Returns (ok, reasons, metrics)."""
    lim, reasons = _limits(cfg), []
    if img is None or img.size == 0:
        return False, ["no image captured"], {}
    m = capture_metrics(img)
    if m["sharpness"] < lim["min_sharpness"]:
        reasons.append(f"image blurred (sharpness {m['sharpness']} < {lim['min_sharpness']}): check focus")
    if m["brightness"] < lim["min_brightness"]:
        reasons.append(f"image too dark (brightness {m['brightness']}): check lighting")
    if m["brightness"] > lim["max_brightness"]:
        reasons.append(f"image too bright (brightness {m['brightness']}): check lighting")
    if m["clipped"] > lim["max_clipped"]:
        reasons.append(f"{m['clipped']:.0%} of pixels over/under-exposed: check lighting/glare")
    return not reasons, reasons, m

