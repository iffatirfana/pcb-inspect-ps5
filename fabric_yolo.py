"""YOLOv8-seg detector for fabric defects (Hole / Knot / Line / Stain).

.onnx models run on onnxruntime + OpenCV only (no PyTorch, no ultralytics), which is what
the Raspberry Pi uses. .pt models load through ultralytics (lazy import) for PC experiments.

detect(bgr) returns detections in the pixel grid of the image passed in:
    [{"type", "confidence", "box": (x1, y1, x2, y2), "polygon": Nx2 float32 array or None}]
fabric_profile.findings_from_detections turns them into mm.

Model layout (ultralytics segment export): output0 = [1, 4 + nc + 32, N] (cx, cy, w, h,
class scores, mask coefficients), output1 = mask prototypes [1, 32, H/4, W/4].
Weights trained with ultralytics (AGPL-3.0) on a public denim dataset (CC BY 4.0);
credit both in the report.
"""
import ast
import os

import cv2
import numpy as np

DEFAULT_NAMES = {0: "Hole", 1: "Knot", 2: "Line", 3: "Stain"}


class YoloDetector:
    def __init__(self, model_path, conf=0.5, imgsz=640, iou=0.4):
        self.conf, self.imgsz, self.iou = conf, imgsz, iou
        self.model_path = str(model_path)
        self._yolo = None
        self._sess = None
        if self.model_path.lower().endswith(".onnx"):
            import onnxruntime as ort
            opts = ort.SessionOptions()
            if os.environ.get("FABRIC_ORT_THREADS"):                 # set by tools/pi_benchmark.py
                opts.intra_op_num_threads = int(os.environ["FABRIC_ORT_THREADS"])
            self._sess = ort.InferenceSession(self.model_path, opts,
                                              providers=["CPUExecutionProvider"])
            shape = self._sess.get_inputs()[0].shape            # [1, 3, H, W] or dynamic
            h, w = shape[2], shape[3]
            self.in_h = h if isinstance(h, int) else int(imgsz)
            self.in_w = w if isinstance(w, int) else int(imgsz)
            names = self._sess.get_modelmeta().custom_metadata_map.get("names")
            try:
                self.names = {int(k): v for k, v in ast.literal_eval(names).items()}
            except Exception:
                self.names = dict(DEFAULT_NAMES)
        else:
            from ultralytics import YOLO
            self._yolo = YOLO(self.model_path, task="segment")

    # ---------------------------------------------------------------- ultralytics path (.pt)
    def _detect_yolo(self, bgr):
        r = self._yolo.predict(source=bgr, imgsz=self.imgsz, conf=self.conf, iou=self.iou,
                               verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0:
            return []
        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        cls = r.boxes.cls.cpu().numpy().astype(int)
        polys = r.masks.xy if r.masks is not None else [None] * len(boxes)
        return [{"type": r.names[int(c)], "confidence": float(s), "box": tuple(map(float, b)),
                 "polygon": p} for b, s, c, p in zip(boxes, confs, cls, polys)]

    # ---------------------------------------------------------------- onnxruntime path
    def _letterbox(self, bgr):
        h, w = bgr.shape[:2]
        r = min(self.in_h / h, self.in_w / w)
        nh, nw = int(round(h * r)), int(round(w * r))
        im = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        top, left = (self.in_h - nh) // 2, (self.in_w - nw) // 2
        out = np.full((self.in_h, self.in_w, 3), 114, np.uint8)
        out[top:top + nh, left:left + nw] = im
        return out, r, left, top

    def _polygon(self, coef, protos, box_in, r, left, top, h, w):
        """Mask polygon in original-image pixels for one detection."""
        ph, pw = protos.shape[1:]
        m = 1.0 / (1.0 + np.exp(-(coef @ protos.reshape(protos.shape[0], -1))))
        m = m.reshape(ph, pw)
        m = cv2.resize(m, (self.in_w, self.in_h), interpolation=cv2.INTER_LINEAR)
        x1, y1, x2, y2 = [int(round(v)) for v in box_in]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(self.in_w, x2), min(self.in_h, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        crop = (m[y1:y2, x1:x2] > 0.5).astype(np.uint8)
        cs, _ = cv2.findContours(crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cs:
            return None
        c = max(cs, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)
        if len(c) < 3:
            return None
        c[:, 0] = (c[:, 0] + x1 - left) / r
        c[:, 1] = (c[:, 1] + y1 - top) / r
        c[:, 0] = np.clip(c[:, 0], 0, w - 1)
        c[:, 1] = np.clip(c[:, 1], 0, h - 1)
        return c

    def _detect_onnx(self, bgr):
        h, w = bgr.shape[:2]
        x, r, left, top = self._letterbox(bgr)
        blob = cv2.cvtColor(x, cv2.COLOR_BGR2RGB).astype(np.float32).transpose(2, 0, 1)[None] / 255.0
        out0, protos = self._sess.run(None, {self._sess.get_inputs()[0].name: blob})[:2]
        o = out0[0].T                                            # [N, 4 + nc + 32]
        nc = len(self.names)
        scores = o[:, 4:4 + nc]
        cls = scores.argmax(1)
        conf = scores.max(1)
        keep = conf >= self.conf
        if not keep.any():
            return []
        o, cls, conf = o[keep], cls[keep], conf[keep]
        xywh = np.stack([o[:, 0] - o[:, 2] / 2, o[:, 1] - o[:, 3] / 2, o[:, 2], o[:, 3]], 1)
        idx = cv2.dnn.NMSBoxesBatched(xywh.tolist(), conf.tolist(), cls.tolist(),
                                      self.conf, self.iou)
        idx = np.array(idx).reshape(-1)
        dets = []
        for i in idx:
            bx, by, bw, bh = xywh[i]
            box_in = (bx, by, bx + bw, by + bh)                  # in letterboxed input pixels
            x1 = float(np.clip((bx - left) / r, 0, w))
            y1 = float(np.clip((by - top) / r, 0, h))
            x2 = float(np.clip((bx + bw - left) / r, 0, w))
            y2 = float(np.clip((by + bh - top) / r, 0, h))
            poly = self._polygon(o[i, 4 + nc:], protos[0], box_in, r, left, top, h, w)
            dets.append({"type": self.names.get(int(cls[i]), str(int(cls[i]))),
                         "confidence": float(conf[i]), "box": (x1, y1, x2, y2),
                         "polygon": poly})
        return dets

    def detect(self, bgr):
        return self._detect_yolo(bgr) if self._yolo is not None else self._detect_onnx(bgr)
