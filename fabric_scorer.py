"""Fabric anomaly scorer: backbone ONNX + normal-feature bank -> heatmap.

Contract with Person 2 (must match):
  preprocessing : gray_world(bgr) from this file is applied to every image BEFORE resizing,
                  for the bank (training) and for scoring. Import it; do not reimplement.
  backbone ONNX : input  float32 (1,3,H,W), RGB, /255, ImageNet mean/std normalised,
                          H,W = scorer.work_size in kb/fabric.yaml
                  output float32 (1,C,h,w), ONE fused feature map (layer2+layer3 already
                          resized to a common grid and concatenated)
  bank .npy     : float32 (N,C) normal features, same C
Heatmap value = distance to the nearest bank feature, upsampled to the crop size.
"""
import cv2
import numpy as np

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def gray_world(bgr, target_mean=128.0):
    """Removes colour-cast and exposure drift between photos: scales each channel so the
    three channel means are equal, then scales the whole image to a fixed mean brightness.
    MUST be applied identically when building the bank (training) and when scoring."""
    f = bgr.astype(np.float32)
    ch = f.reshape(-1, 3).mean(0)
    f *= (ch.mean() / np.maximum(ch, 1e-6))[None, None, :]
    f *= target_mean / max(float(f.mean()), 1e-6)
    return np.clip(f, 0, 255).astype(np.uint8)


class FabricScorer:
    def __init__(self, onnx_path, bank_path, work_size=(512, 512), blur_ksize=15):
        import onnxruntime as ort          # imported here so tests can run without it
        self.sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name
        self.bank = np.load(bank_path).astype(np.float32)
        self.bank_sq = (self.bank ** 2).sum(1)
        self.work_size = tuple(work_size)
        self.blur = int(blur_ksize) | 1

    def heatmap(self, bgr):
        h0, w0 = bgr.shape[:2]
        rgb = cv2.cvtColor(cv2.resize(gray_world(bgr), self.work_size), cv2.COLOR_BGR2RGB)
        x = ((rgb.astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1)[None]
        feat = self.sess.run(None, {self.inp: x})[0][0]          # (C,h,w)
        c, h, w = feat.shape
        if c != self.bank.shape[1]:
            raise ValueError(f"feature size {c} != bank size {self.bank.shape[1]}")
        f = feat.reshape(c, -1).T                                # (hw,C)
        f_sq = (f ** 2).sum(1)
        out = np.empty(len(f), np.float32)
        for i in range(0, len(f), 512):                          # chunk: keeps RAM low on the Pi
            blk = f[i:i + 512]
            d = f_sq[i:i + 512, None] - 2.0 * blk @ self.bank.T + self.bank_sq[None]
            out[i:i + 512] = np.sqrt(np.maximum(d.min(1), 0.0))
        hm = cv2.GaussianBlur(out.reshape(h, w), (self.blur, self.blur), 0)
        return cv2.resize(hm, (w0, h0), interpolation=cv2.INTER_LINEAR)
