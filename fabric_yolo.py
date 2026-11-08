"""YOLOv8-seg detector for fabric defects (Hole / Knot / Line / Stain).

Loads a .pt or .onnx model through ultralytics (lazy import, so the rest of the
pipeline and the tests run without it). Returns detections in the pixel grid of
the image passed in; fabric_profile.findings_from_detections turns them into mm.

Needs ultralytics (AGPL-3.0). Credit the public training dataset in the report.
"""


class YoloDetector:
    def __init__(self, model_path, conf=0.5, imgsz=640, iou=0.4):
        from ultralytics import YOLO
        self.model = YOLO(model_path, task="segment")
        self.conf, self.imgsz, self.iou = conf, imgsz, iou

    def detect(self, bgr):
        r = self.model.predict(source=bgr, imgsz=self.imgsz, conf=self.conf, iou=self.iou,
                               verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0:
            return []
        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        cls = r.boxes.cls.cpu().numpy().astype(int)
        polys = r.masks.xy if r.masks is not None else [None] * len(boxes)
        return [{"type": r.names[int(c)], "confidence": float(s), "box": tuple(map(float, b)),
                 "polygon": p} for b, s, c, p in zip(boxes, confs, cls, polys)]
