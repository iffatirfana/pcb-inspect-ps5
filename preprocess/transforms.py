"""Shared image preprocessing for every model (training AND inference).

Rule: never resize/normalise images anywhere else - always import from here,
so Kaggle/Colab training and the Raspberry Pi see identical inputs.
"""
from pathlib import Path

import cv2
import numpy as np
import yaml

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_config(path="config.yaml"):
    """Read the shared config.yaml."""
    with open(Path(path), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize_lighting(img_bgr, clip=2.0, grid=(8, 8)):
    """Even out brightness with CLAHE on the L channel (keeps colours)."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=tuple(grid))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def to_model_input(img_bgr, size):
    """BGR uint8 image -> float32 NCHW tensor, ImageNet-normalised.

    size: [width, height] from config.yaml (e.g. anomaly.input_size).
    Returns array of shape (1, 3, H, W).
    """
    img = cv2.resize(img_bgr, tuple(size), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return img.transpose(2, 0, 1)[None].astype(np.float32)


def crop(img_bgr, bbox, pad=8):
    """Cut a defect region [x1, y1, x2, y2] out of the image, with padding."""
    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = bbox
    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
    x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
    return img_bgr[y1:y2, x1:x2]
