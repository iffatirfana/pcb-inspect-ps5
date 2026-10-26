"""Inspectors: one per board type, all returning the same findings format.

    findings = get_inspector(profile, cfg).detect(aligned_bgr, valid_mask)
    finding  = {type, conf, bbox:[x1,y1,x2,y2], area_px, source}

  assembled -> GoldenDiffInspector  golden-diff boxes, named by the MobileNetV3
                                    classifier (no classifier yet -> "unknown_anomaly")
  bare      -> YoloInspector        YOLO ONNX, tiled over the full image
  demo      -> DemoInspector        fixed example findings (tests, laptop demo)

Models are ONNX files in models/ (shared via Hugging Face), run with
onnxruntime on the Pi CPU. Hailo versions can replace them later with the
same inputs/outputs.
"""
from pathlib import Path

import cv2
import numpy as np

from preprocess.transforms import IMAGENET_MEAN, IMAGENET_STD, tile_image


class InspectorError(RuntimeError):
    pass


def _session(path):
    import onnxruntime as ort
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


# ------------------------------------------------------------------ demo
class DemoInspector:
    name = "demo"
    FINDINGS = [
        {"type": "solder_bridge", "conf": 0.94, "bbox": [348, 292, 356, 308], "area_px": 128},
        {"type": "tombstone", "conf": 0.88, "bbox": [680, 190, 720, 210], "area_px": 800},
    ]

    def detect(self, img, valid=None):
        return [{**f, "source": self.name} for f in self.FINDINGS]


# ------------------------------------------------------------------ classifier
class Classifier:
    """MobileNetV3 crop classifier (ONNX). Input 1x3xHxW ImageNet-normalised RGB,
    output logits over `classes`. Class 'normal' = false alarm (finding dropped)."""

    def __init__(self, model, classes, input_size=(128, 128), min_conf=0.6, context=0.5):
        self.classes, self.size = list(classes), tuple(input_size)
        self.min_conf, self.context = min_conf, context
        self.sess = _session(model) if model and Path(model).exists() else None

    @property
    def available(self):
        return self.sess is not None

    def crop(self, img, bbox):
        """Square crop around the box with surrounding context."""
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        side = max(x2 - x1, y2 - y1) * (1 + 2 * self.context)
        side = max(side, 32)
        h, w = img.shape[:2]
        a, b = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
        c, d = int(min(w, cx + side / 2)), int(min(h, cy + side / 2))
        return img[b:d, a:c]

    def classify(self, img, bbox):
        """Returns (type, conf). No model -> ('unknown_anomaly', None)."""
        if not self.available:
            return "unknown_anomaly", None
        c = cv2.resize(self.crop(img, bbox), self.size, interpolation=cv2.INTER_AREA)
        x = (cv2.cvtColor(c, cv2.COLOR_BGR2RGB).astype(np.float32) / 255 - IMAGENET_MEAN) / IMAGENET_STD
        logits = self.sess.run(None, {self.sess.get_inputs()[0].name:
                                      x.transpose(2, 0, 1)[None].astype(np.float32)})[0][0]
        p = np.exp(logits - logits.max())
        p /= p.sum()
        i = int(p.argmax())
        conf = round(float(p[i]), 3)
        if conf < self.min_conf:
            return "unknown_anomaly", conf
        return self.classes[i], conf


# ------------------------------------------------------------------ assembled
class GoldenDiffInspector:
    name = "golden_diff"

    def __init__(self, profile, cfg):
        from vision.golden_diff import GoldenDiff
        a, c = cfg.get("anomaly", {}), cfg.get("classifier", {})
        if not profile.anomaly_threshold:
            raise InspectorError(f"board '{profile.id}': anomaly_threshold not set")
        self.gd = GoldenDiff(tile=a.get("tile_size", 256), overlap=a.get("tile_overlap", 32),
                             onnx=a.get("backbone_onnx"), work_width=a.get("work_width"))
        golden = cv2.imread(str(profile.golden_path))
        refs = [golden]
        for p in profile.references:
            img = cv2.imread(str(p))
            if img is not None:
                refs.append(profile.aligner.align(img)[0])
        self.gd.set_references(refs)
        self.gd.threshold = float(profile.anomaly_threshold)
        self.min_area = a.get("min_blob_area_px", 30)
        self.classifier = Classifier(c.get("model"), c.get("classes", []) + ["normal"],
                                     c.get("input_size", (128, 128)), c.get("min_conf", 0.6))

    def detect(self, img, valid=None):
        out = []
        for box in self.gd.detect(self.gd.score(img, valid), min_area=self.min_area):
            kind, cconf = self.classifier.classify(img, box["bbox"])
            if kind == "normal":
                continue                                   # classifier: false alarm
            out.append({**box, "type": kind, "anomaly_conf": box["conf"],
                        "conf": cconf if cconf is not None else box["conf"],
                        "source": self.name})
        return out


# ------------------------------------------------------------------ bare
def decode_yolo(out, classes, conf_thr=0.25, offset=(0, 0), scale=1.0):
    """YOLOv8/11 ONNX output (1, 4+nc, N) -> [(x1,y1,x2,y2), conf, class]."""
    p = out[0].T                                          # N, 4+nc
    scores = p[:, 4:]
    cls = scores.argmax(1)
    conf = scores[np.arange(len(p)), cls]
    keep = conf >= conf_thr
    res = []
    for (cx, cy, w, h), c, k in zip(p[keep, :4], conf[keep], cls[keep]):
        x1, y1 = (cx - w / 2) / scale + offset[0], (cy - h / 2) / scale + offset[1]
        x2, y2 = (cx + w / 2) / scale + offset[0], (cy + h / 2) / scale + offset[1]
        res.append(((x1, y1, x2, y2), float(c), classes[int(k)]))
    return res


def nms(dets, iou=0.5):
    if not dets:
        return []
    boxes = [[b[0], b[1], b[2] - b[0], b[3] - b[1]] for b, _, _ in dets]
    idx = cv2.dnn.NMSBoxes(boxes, [c for _, c, _ in dets], 0.0, iou)
    return [dets[i] for i in np.array(idx).flatten()]


class YoloInspector:
    name = "yolo"

    def __init__(self, profile, cfg):
        y = cfg.get("yolo", {})
        model = y.get("model", "models/yolo_bare.onnx")
        if not Path(model).exists():
            raise InspectorError(f"YOLO model not found: {model} (Person 2: train + export)")
        self.sess = _session(model)
        self.inp = self.sess.get_inputs()[0].name
        self.size = int(y.get("imgsz", 640))
        self.classes = y.get("classes", [])
        self.conf, self.iou = y.get("conf", 0.25), y.get("iou", 0.5)
        self.overlap = y.get("tile_overlap", 64)

    def detect(self, img, valid=None):
        h, w = img.shape[:2]
        if h < self.size or w < self.size:                # pad small images up to one tile
            img = cv2.copyMakeBorder(img, 0, max(0, self.size - h), 0, max(0, self.size - w),
                                     cv2.BORDER_CONSTANT)
        dets = []
        for x, y, t in tile_image(img, self.size, self.overlap):
            blob = cv2.cvtColor(t, cv2.COLOR_BGR2RGB).astype(np.float32).transpose(2, 0, 1)[None] / 255
            out = self.sess.run(None, {self.inp: blob})[0]
            dets += decode_yolo(out, self.classes, self.conf, offset=(x, y))
        res = []
        for (x1, y1, x2, y2), c, k in nms(dets, self.iou):
            x1, y1, x2, y2 = (int(round(v)) for v in (max(0, x1), max(0, y1), min(w, x2), min(h, y2)))
            if valid is not None and not valid[min(h - 1, (y1 + y2) // 2), min(w - 1, (x1 + x2) // 2)]:
                continue
            res.append({"type": k, "conf": round(c, 3), "bbox": [x1, y1, x2, y2],
                        "area_px": (x2 - x1) * (y2 - y1), "source": self.name})
        return res


# ------------------------------------------------------------------ factory
_CACHE = {}


def get_inspector(profile, cfg):
    """Build (once) and return the inspector for this board profile."""
    if profile.id not in _CACHE:
        kind = profile.inspector
        if kind == "demo":
            _CACHE[profile.id] = DemoInspector()
        elif kind == "golden_diff":
            _CACHE[profile.id] = GoldenDiffInspector(profile, cfg)
        elif kind == "yolo":
            _CACHE[profile.id] = YoloInspector(profile, cfg)
        else:
            raise InspectorError(f"unknown inspector '{kind}' for board '{profile.id}'")
    return _CACHE[profile.id]
