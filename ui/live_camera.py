"""Live camera for the dashboard: one background thread keeps the webcam open, keeps the newest frame,
and serves it as an MJPEG stream. A click on 'Capture & inspect' uses the newest full-size frame, so the
photo that is inspected is the one the operator is looking at.

The camera is only open while the live view is on, so other tools (shoot.py, the Capture button) can use it
after Stop.
"""
import sys
import threading
import time

import cv2


def _set_size(cap, mjpg):
    if mjpg:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    return cap


def candidates(index=0):
    """Ways to open a USB webcam, best first. Opened lazily: the next one is tried only if the previous gives
    no picture (some cameras open fine but send black frames with one backend)."""
    if sys.platform.startswith("win"):
        for api, mjpg in ((cv2.CAP_MSMF, False), (cv2.CAP_DSHOW, False), (cv2.CAP_DSHOW, True)):
            yield _set_size(cv2.VideoCapture(index, api), mjpg)
    elif sys.platform.startswith("linux"):
        yield _set_size(cv2.VideoCapture(index, cv2.CAP_V4L2), True)
        yield _set_size(cv2.VideoCapture(index), False)
    else:
        yield _set_size(cv2.VideoCapture(index), False)


def open_capture(index=0):
    """First candidate that opens (kept for callers that want a plain capture)."""
    for cap in candidates(index):
        if cap.isOpened():
            return cap
        cap.release()
    return cv2.VideoCapture(index)


def probe(cap, tries=30, discard=6, min_mean=3.0):
    """Read a few frames; True when a real (not black) picture arrives. Returns (ok, frame)."""
    last = None
    for n in range(tries):
        ok, f = cap.read()
        if not ok or f is None:
            time.sleep(0.05)
            continue
        if n >= discard and float(f.mean()) >= min_mean:
            return True, f
        last = f
        time.sleep(0.03)
    return False, last


class LiveCamera:
    def __init__(self, index=0, opener=None, stream_width=960, fps=15):
        self.index, self.opener = index, opener
        self.stream_width, self.delay = stream_width, 1.0 / fps
        self._lock = threading.Lock()
        self._frame = None
        self._jpeg = None
        self._stop = threading.Event()
        self._thread = None
        self.error = None
        self.size = None

    @property
    def running(self):
        return self._thread is not None and self._thread.is_alive()

    def start(self):
        if self.running:
            return
        self.error, self._frame, self._jpeg = None, None, None
        self._stop.clear()
        gen = iter([self.opener(self.index)]) if self.opener else candidates(self.index)
        opened = black = False
        for cap in gen:
            if not cap.isOpened():
                cap.release()
                continue
            opened = True
            ok, first = probe(cap)
            if ok:
                self._thread = threading.Thread(target=self._loop, args=(cap, first), daemon=True)
                self._thread.start()
                return
            black = True
            cap.release()
        if not opened:
            self.error = "Camera not found. Close other programs that use it, then try again."
        elif black:
            self.error = ("The camera opens but sends a black picture. Close other programs that use it (Camera app, "
                          "Teams, a shoot.py window), remove any lens cover, or unplug and replug it, then press Start again.")

    def _loop(self, cap, first=None):
        bad = 0
        try:
            while not self._stop.is_set():
                if first is not None:
                    ok, frame, first = True, first, None
                else:
                    ok, frame = cap.read()
                if not ok or frame is None:
                    bad += 1
                    if bad > 30:
                        self.error = "The camera stopped sending frames."
                        return
                    time.sleep(0.05)
                    continue
                bad = 0
                h, w = frame.shape[:2]
                small = frame if w <= self.stream_width else cv2.resize(frame, (self.stream_width, int(h * self.stream_width / w)))
                ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 70])
                with self._lock:
                    self._frame, self.size = frame, (w, h)
                    if ok:
                        self._jpeg = buf.tobytes()
                time.sleep(self.delay)
        finally:
            cap.release()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3)
        self._thread = None

    def snapshot(self):
        """Newest full-size frame (BGR copy) or None."""
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def wait_frame(self, timeout=3.0):
        t = time.time()
        while time.time() - t < timeout:
            with self._lock:
                if self._jpeg is not None:
                    return True
            time.sleep(0.05)
        return False

    def mjpeg(self):
        """Generator for a multipart/x-mixed-replace response. Ends when the live view is stopped."""
        last = None
        while self.running and not self._stop.is_set():
            with self._lock:
                jpg = self._jpeg
            if jpg is not None and jpg is not last:
                last = jpg
                yield b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(jpg) + jpg + b"\r\n"
            time.sleep(self.delay / 2)
