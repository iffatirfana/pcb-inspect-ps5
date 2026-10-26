"""Golden-feature difference: one anomaly model for every assembled board type.

A frozen ImageNet ResNet18 (layer2 + layer3) turns the aligned test image and
the board's good reference image(s) into feature maps, tile by tile. Each spot's
anomaly score is the cosine distance to the closest reference feature within a
small neighbourhood (tolerates 1-2 px misalignment). No training per board: a
new board needs its golden photo (+ a few good photos = fewer false alarms).

Feature extractor, whichever is available:
  * ONNX file (Pi: onnxruntime, no PyTorch)  -> GoldenDiff(onnx="models/golden_diff_resnet18.onnx")
  * PyTorch + torchvision (Kaggle / laptop)  -> GoldenDiff()
Make the ONNX once on Kaggle/laptop:  GoldenDiff().export_onnx()

    gd = GoldenDiff(onnx=...)
    gd.set_references([golden_bgr, good2_bgr, ...])   # already aligned to golden
    gd.calibrate([good_a, good_b, ...])               # sets threshold
    heat = gd.score(aligned_test_bgr, valid)          # float32 HxW
    boxes = gd.detect(heat)                           # [{type, conf, bbox, area_px}]
"""
from pathlib import Path

import cv2
import numpy as np

from preprocess.transforms import IMAGENET_MEAN, IMAGENET_STD, stitch_maps, tile_image


def _torch_backbone(pretrained=True):
    import torch
    import torch.nn.functional as F
    import torchvision

    class Backbone(torch.nn.Module):
        """ResNet18 layer2 + layer3 (upsampled), concatenated, 3x3 average."""

        def __init__(self):
            super().__init__()
            w = torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None
            r = torchvision.models.resnet18(weights=w).eval()
            self.stem = torch.nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool, r.layer1)
            self.layer2, self.layer3 = r.layer2, r.layer3

        def forward(self, x):                  # N,3,H,W normalised -> N,384,H/8,W/8
            f2 = self.layer2(self.stem(x))
            f3 = self.layer3(f2)
            f3 = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
            return F.avg_pool2d(torch.cat([f2, f3], 1), 3, 1, 1)

    return Backbone().eval()


class GoldenDiff:
    def __init__(self, tile=256, overlap=32, search=1, onnx=None, pretrained=True,
                 work_width=None, batch=8, extractor=None):
        """work_width: process at this image width (px) and scale results back;
        keeps memory/time bounded for 4056-px camera images (None = full size)."""
        self.tile, self.overlap, self.search = tile, overlap, search
        self.work_width, self.batch = work_width, batch
        self.threshold = None
        self.refs = None
        self._net = self._sess = None
        if extractor is not None:              # tests: any fn(N,3,H,W float32) -> N,C,h,w
            self._extract = extractor
        elif onnx and Path(onnx).exists():
            import onnxruntime as ort
            self._sess = ort.InferenceSession(str(onnx), providers=["CPUExecutionProvider"])
            self._extract = lambda x: self._sess.run(None, {"image": x})[0]
        else:
            import torch
            self._net = _torch_backbone(pretrained)
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._net.to(self._device)

            def run(x):
                with torch.no_grad():
                    return self._net(torch.from_numpy(x).to(self._device)).cpu().numpy()
            self._extract = run

    # ---------- helpers ----------
    def _work(self, img):
        """Resize to working width; returns (image, scale)."""
        if not self.work_width or img.shape[1] <= self.work_width:
            return img, 1.0
        s = self.work_width / img.shape[1]
        return cv2.resize(img, (self.work_width, round(img.shape[0] * s)),
                          interpolation=cv2.INTER_AREA), s

    def _features(self, tiles):
        out = []
        for i in range(0, len(tiles), self.batch):
            x = np.stack([cv2.cvtColor(t, cv2.COLOR_BGR2RGB) for t in tiles[i:i + self.batch]])
            x = ((x.astype(np.float32) / 255 - IMAGENET_MEAN) / IMAGENET_STD)
            f = self._extract(np.ascontiguousarray(x.transpose(0, 3, 1, 2), np.float32))
            f /= np.linalg.norm(f, axis=1, keepdims=True) + 1e-6
            out.append(f.astype(np.float16))
        return np.concatenate(out)                             # T,C,h,w unit vectors

    # ---------- references ----------
    def set_references(self, good_images):
        """good_images: BGR images, all aligned to the golden image (golden first)."""
        work = [self._work(g)[0] for g in good_images]
        self.shape = good_images[0].shape[:2]
        self.work_shape = work[0].shape[:2]
        self.positions = [(x, y) for x, y, _ in tile_image(work[0], self.tile, self.overlap)]
        per_img = [self._features([t for _, _, t in tile_image(w, self.tile, self.overlap)])
                   for w in work]
        self.refs = np.stack(per_img, 1)                       # T,R,C,h,w (float16)

    # ---------- scoring ----------
    def _tile_score(self, ref, feat):
        """ref R,C,h,w ; feat C,h,w -> h,w distance (min over refs and shifts)."""
        s = self.search
        _, _, h, w = ref.shape
        pad = np.pad(ref.astype(np.float32), ((0, 0), (0, 0), (s, s), (s, s)), mode="edge")
        f = feat.astype(np.float32)
        best = np.full((h, w), 2.0, np.float32)
        for dy in range(2 * s + 1):
            for dx in range(2 * s + 1):
                sim = np.einsum("rchw,chw->rhw", pad[:, :, dy:dy + h, dx:dx + w], f)
                best = np.minimum(best, (1 - sim).min(0))
        return best

    def score(self, img, valid=None):
        """Anomaly heatmap at the input image size. valid: bool mask (see valid_mask)."""
        if self.refs is None:
            raise RuntimeError("call set_references() first")
        if img.shape[:2] != self.shape:
            img = cv2.resize(img, (self.shape[1], self.shape[0]))
        work, _ = self._work(img)
        tiles = tile_image(work, self.tile, self.overlap)
        feats = self._features([t for _, _, t in tiles])
        maps = [self._tile_score(self.refs[i], feats[i]) for i in range(len(tiles))]
        heat = stitch_maps([(x, y, m) for (x, y, _), m in zip(tiles, maps)],
                           work.shape[:2], self.tile)
        heat = cv2.GaussianBlur(heat, (0, 0), 4)
        if heat.shape != tuple(self.shape):
            heat = cv2.resize(heat, (self.shape[1], self.shape[0]), interpolation=cv2.INTER_LINEAR)
        if valid is not None:
            heat[~valid] = 0
        return heat

    @staticmethod
    def valid_mask(H, test_shape, golden_shape, border=24):
        """Pixels of the aligned image that came from the camera (not warp fill)."""
        ones = np.full(test_shape[:2], 255, np.uint8)
        m = cv2.warpPerspective(ones, H, (golden_shape[1], golden_shape[0]))
        return cv2.erode(m, np.ones((border, border), np.uint8)) > 0

    # ---------- threshold + boxes ----------
    def calibrate(self, good_images, valids=None, margin=1.1):
        """Threshold = highest score seen on good boards x margin."""
        valids = valids or [None] * len(good_images)
        self.threshold = margin * max(float(self.score(g, v).max())
                                      for g, v in zip(good_images, valids))
        return self.threshold

    def detect(self, heat, threshold=None, min_area=30):
        thr = threshold or self.threshold
        if thr is None:
            raise RuntimeError("no threshold: call calibrate() or pass threshold")
        mask = (heat > thr).astype(np.uint8)
        n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
        out = []
        for i in range(1, n):
            x, y, w, h, area = (int(v) for v in stats[i])
            if area < min_area:
                continue
            peak = float(heat[y:y + h, x:x + w].max())
            out.append({"type": "anomaly", "conf": round(min(1.0, peak / (2 * thr)), 3),
                        "bbox": [x, y, x + w, y + h], "area_px": area})
        return out

    # ---------- export (Pi: onnxruntime; Hailo: compile this ONNX) ----------
    def export_onnx(self, path="models/golden_diff_resnet18.onnx"):
        import torch
        net = self._net if self._net is not None else _torch_backbone(True)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.onnx.export(net.cpu(), torch.zeros(1, 3, self.tile, self.tile), path,
                          input_names=["image"], output_names=["features"],
                          dynamic_axes={"image": {0: "n"}, "features": {0: "n"}},
                          opset_version=17, dynamo=False)
        return path
