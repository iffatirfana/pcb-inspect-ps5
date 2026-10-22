"""Stage 2: align a test photo to the golden (known-good) board image.

ORB keypoints + brute-force matching + RANSAC homography. Returns the
warped test image (same size/pose as golden) and the homography, so every
pixel of the test board lines up with the golden board and the board map.
"""
import cv2
import numpy as np


class AlignError(RuntimeError):
    pass


class Aligner:
    def __init__(self, golden_bgr, n_features=5000, ransac_thresh=3.0,
                 min_inliers=30, ratio=0.75):
        self.golden = golden_bgr
        self.h, self.w = golden_bgr.shape[:2]
        self.orb = cv2.ORB_create(nfeatures=n_features)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.ransac_thresh = ransac_thresh
        self.min_inliers = min_inliers
        self.ratio = ratio
        self.kp_g, self.des_g = self.orb.detectAndCompute(self._gray(golden_bgr), None)
        if self.des_g is None or len(self.kp_g) < min_inliers:
            raise AlignError("Golden image has too few features - check focus/lighting")

    @staticmethod
    def _gray(img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img

    def align(self, test_bgr):
        """Return (aligned_bgr, H, stats). H maps test pixels -> golden pixels."""
        kp_t, des_t = self.orb.detectAndCompute(self._gray(test_bgr), None)
        if des_t is None or len(kp_t) < self.min_inliers:
            raise AlignError("Test image has too few features")
        pairs = self.matcher.knnMatch(des_t, self.des_g, k=2)
        good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < self.ratio * n.distance]
        if len(good) < self.min_inliers:
            raise AlignError(f"Only {len(good)} good matches")
        src = np.float32([kp_t[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([self.kp_g[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, self.ransac_thresh)
        if H is None:
            raise AlignError("Homography failed")
        inliers = int(mask.sum())
        if inliers < self.min_inliers:
            raise AlignError(f"Only {inliers} RANSAC inliers")
        aligned = cv2.warpPerspective(test_bgr, H, (self.w, self.h))
        err = np.linalg.norm(cv2.perspectiveTransform(src[mask.ravel() == 1], H)
                             - dst[mask.ravel() == 1], axis=2).mean()
        return aligned, H, {"matches": len(good), "inliers": inliers,
                            "reproj_err_px": round(float(err), 2)}


def synthetic_board(w=800, h=600, seed=1):
    """Fake PCB: green board, copper pads, traces, IC bodies, text."""
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), (40, 110, 30), np.uint8)
    for _ in range(60):
        x, y = int(rng.integers(20, w - 60)), int(rng.integers(20, h - 60))
        cv2.line(img, (x, y), (x + int(rng.integers(-80, 80)), y + int(rng.integers(-80, 80))),
                 (60, 150, 190), 2)
    for _ in range(12):
        x, y = int(rng.integers(40, w - 120)), int(rng.integers(40, h - 120))
        cv2.rectangle(img, (x, y), (x + 70, y + 50), (25, 25, 25), -1)
        for i in range(6):
            cv2.rectangle(img, (x - 8, y + 4 + i * 8), (x, y + 8 + i * 8), (170, 170, 175), -1)
            cv2.rectangle(img, (x + 70, y + 4 + i * 8), (x + 78, y + 8 + i * 8), (170, 170, 175), -1)
    for i in range(10):
        cv2.putText(img, f"R{i+1}", (int(rng.integers(10, w - 50)), int(rng.integers(20, h - 10))),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1)
    return img


def self_test():
    golden = synthetic_board()
    h, w = golden.shape[:2]
    # simulate board placed 12 px right, 7 px down, rotated 2 degrees, slight perspective
    M = cv2.getRotationMatrix2D((w / 2, h / 2), 2.0, 1.0)
    M[0, 2] += 12
    M[1, 2] += 7
    H_true = np.vstack([M, [0.00001, -0.00002, 1]])
    test = cv2.warpPerspective(golden, H_true, (w, h))
    test = cv2.add(test, np.full_like(test, 15))            # brighter lighting
    aligned, H, stats = Aligner(golden).align(test)
    margin = 40
    diff = cv2.absdiff(aligned, cv2.add(golden, np.full_like(golden, 15)))[margin:-margin, margin:-margin]
    print("stats:", stats)
    print("mean residual difference after alignment (0-255):", round(float(diff.mean()), 2))
    corner_err = np.linalg.norm(cv2.perspectiveTransform(
        np.float32([[[0, 0]], [[w, 0]], [[0, h]], [[w, h]]]), H @ H_true)
        - np.float32([[[0, 0]], [[w, 0]], [[0, h]], [[w, h]]]), axis=2).max()
    print("max corner error vs true pose (px):", round(float(corner_err), 2))
    cv2.imwrite("results/align_golden.png", golden)
    cv2.imwrite("results/align_test.png", test)
    cv2.imwrite("results/align_aligned.png", aligned)
    print("saved results/align_*.png")


if __name__ == "__main__":
    self_test()
