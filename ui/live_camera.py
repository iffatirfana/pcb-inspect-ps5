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


def open_capture(index=0):
    """Open a USB webcam the way that worked on our machines: Windows MSMF without MJPG (DirectShow as a
    fallback), Linux V4L2 with MJPG 1920x1080."""
    if sys.platform.startswith("win"):
        for api in (cv2.CAP_MSMF, cv2.CAP_DSHOW):
            cap = cv2.VideoCapture(index, api)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
                return cap
            cap.release()
        return cv2.VideoCapture(index)
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2) if sys.platform.startswith("linux") else cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    return cap


class LiveCamera:
    def __init__(self, index=0, opener=None, stream_width=960, fps=15):
        self.index, self.opener = index, opener or open_capture
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
        cap = self.opener(self.index)
        if not cap.isOpened():
            cap.release()
            self.error = "Camera not found. Close other programs that use it, then try again."
            return
        self._thread = threading.Thread(target=self._loop, args=(cap,), daemon=True)
        self._thread.start()

    def _loop(self, cap):
        bad = 0
        try:
            while not self._stop.is_set():
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
