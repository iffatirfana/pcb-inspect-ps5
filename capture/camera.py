"""Stage 0: capture a board image.

On the Raspberry Pi: Picamera2 (HQ Camera), fixed exposure/white balance so
every shot is lit the same. Anywhere else (or if no camera): read from a file.
"""
import time
from pathlib import Path

import cv2


class Camera:
    def __init__(self, width=4056, height=3040, fallback_image=None,
                 exposure_us=None, gain=None, awb_gains=None):
        self.fallback = Path(fallback_image) if fallback_image else None
        self.cam = None
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
            time.sleep(1.0)  # let exposure settle
        except Exception as e:  # no Picamera2 / no camera attached
            self.cam = None
            self.reason = f"camera unavailable ({e.__class__.__name__}); using file"

    @property
    def source(self):
        return "picamera2" if self.cam else f"file:{self.fallback}"

    def capture(self):
        if self.cam:
            return self.cam.capture_array("main")  # RGB888 array is BGR order for OpenCV
        if self.fallback and self.fallback.exists():
            img = cv2.imread(str(self.fallback))
            if img is not None:
                return img
        raise RuntimeError("No camera and no fallback image available")

    def close(self):
        if self.cam:
            self.cam.stop()
            self.cam.close()


if __name__ == "__main__":
    cam = Camera(fallback_image="kb/board_sample/test_board.png")
    img = cam.capture()
    print("source:", cam.source, "| shape:", img.shape)
    Path("results").mkdir(exist_ok=True)
    cv2.imwrite("results/capture_test.png", img)
    print("saved results/capture_test.png")
    cam.close()
