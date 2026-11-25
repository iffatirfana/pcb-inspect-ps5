"""Stage 0: capture a fabric image.

source: auto    Picamera2 (HQ Camera) if present, else a file (default)
        picamera2 Picamera2 only
        usb       USB/UVC webcam through OpenCV (V4L2); autofocus off, exposure
                  fixed if usb_exposure is set, white balance fixed only if
                  usb_lock_wb is true
        file      always read the fallback image

Fixed exposure/white balance so every shot is lit the same.
"""
import time
from pathlib import Path

import cv2


class Camera:
    def __init__(self, width=4056, height=3040, fallback_image=None,
                 exposure_us=None, gain=None, awb_gains=None,
                 source="auto", usb_index=0, usb_size=(640, 480), usb_exposure=None,
                 usb_lock_wb=False):
        self.fallback = Path(fallback_image) if fallback_image else None
        self.cam = None
        self.kind = None
        self.reason = ""
        if source == "usb":
            self._open_usb(usb_index, usb_size, usb_exposure, usb_lock_wb)
            return
        if source == "file":
            self.reason = "source=file"
            return
        try:
            from picamera2 import Picamera2  # only installed on the Pi
            self.cam = Picamera2()
            cfg = self.cam.create_still_configuration(main={"size": (width, height),
                                                            "format": "RGB888"})
            self.cam.configure(cfg)
            controls = {}
            if exposure_us:
                controls.update(AeEnable=False, ExposureTime=int(exposure_us))
            if gain:
                controls["AnalogueGain"] = float(gain)
            if awb_gains:
                controls.update(AwbEnable=False, ColourGains=tuple(awb_gains))
            if controls:
                self.cam.set_controls(controls)
            self.cam.start()
            self.kind = "picamera2"
            time.sleep(1.0)  # let exposure settle
        except Exception as e:  # no Picamera2 / no camera attached
            self.cam = None
            self.reason = f"camera unavailable ({e.__class__.__name__}); using file"

    def _open_usb(self, index, size, exposure, lock_wb=False):
        try:
            cap = cv2.VideoCapture(int(index), cv2.CAP_V4L2)
            if not cap.isOpened():
                cap.release()
                raise RuntimeError(f"cannot open /dev/video{index}")
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(size[0]))
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(size[1]))
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)       # ignored by cameras without these
            if lock_wb:                              # off by default: many webcams lock a wrong tint
                cap.set(cv2.CAP_PROP_AUTO_WB, 0)
            if exposure is not None:
                cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)   # V4L2: 1 = manual
                cap.set(cv2.CAP_PROP_EXPOSURE, float(exposure))
            for _ in range(10):                       # let the sensor settle
                cap.read()
            self.cam, self.kind = cap, "usb"
        except Exception as e:
            self.cam = None
            self.reason = f"usb camera unavailable ({e.__class__.__name__}: {e}); using file"

    @property
    def source(self):
        if self.cam:
            return self.kind
        return f"file:{self.fallback}"

    def capture(self):
        if self.cam and self.kind == "usb":
            for _ in range(3):                        # drop buffered frames, keep the newest
                self.cam.grab()
            ok, frame = self.cam.read()
            if not ok or frame is None:
                raise RuntimeError("USB camera returned no frame")
            return frame
        if self.cam:
            return self.cam.capture_array("main")  # RGB888 array is BGR order for OpenCV
        if self.fallback and self.fallback.exists():
            img = cv2.imread(str(self.fallback))
            if img is not None:
                return img
        raise RuntimeError("No camera and no fallback image available")

    def close(self):
        if self.cam and self.kind == "usb":
            self.cam.release()
        elif self.cam:
            self.cam.stop()
            self.cam.close()


if __name__ == "__main__":
    import yaml
    c = yaml.safe_load(open("config.yaml", encoding="utf-8")).get("camera", {})
    cam = Camera(width=c.get("width", 4056), height=c.get("height", 3040),
                 source=c.get("source", "auto"), usb_index=c.get("usb_index", 0),
                 usb_size=tuple(c.get("usb_size", (640, 480))),
                 usb_exposure=c.get("usb_exposure"),
                 usb_lock_wb=c.get("usb_lock_wb", False))
    img = cam.capture()
    print("source:", cam.source, "| shape:", img.shape)
    if cam.reason:
        print("note:", cam.reason)
    Path("results").mkdir(exist_ok=True)
    cv2.imwrite("results/capture_test.png", img)
    print("saved results/capture_test.png")
    cam.close()
