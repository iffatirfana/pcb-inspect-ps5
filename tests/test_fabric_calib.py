import cv2
import numpy as np

import fabric_calib as fc


def tilted_sheet():
    """A 210x297 mm sheet seen at an angle, as the camera would see it."""
    img = np.full((1080, 1920, 3), 90, np.uint8)
    quad = np.array([[700, 200], [1250, 200], [1500, 900], [450, 900]], np.int32)   # TL TR BR BL
    cv2.fillPoly(img, [quad], (255, 255, 255))
    return img, quad.astype(np.float32)


def test_sheet_becomes_a_rectangle_of_the_right_size():
    img, quad = tilted_sheet()
    cal = fc.build_calibration(quad, (210, 297), 3.0, (1920, 1080))
    h = np.array(cal["homography"])
    pts = cv2.perspectiveTransform(quad.reshape(1, 4, 2), h)[0]
    w_px = np.linalg.norm(pts[1] - pts[0])
    h_px = np.linalg.norm(pts[3] - pts[0])
    assert abs(w_px - 210 * 3.0) < 1.5 and abs(h_px - 297 * 3.0) < 1.5
    assert cal["calibrated"] and cal["px_per_mm"] == 3.0
    out = fc.apply(img, cal)
    assert out.shape[1] == cal["warp_size"][0] and out.shape[0] == cal["warp_size"][1]


def test_scale_is_the_same_at_top_and_bottom():
    img, quad = tilted_sheet()
    cal = fc.build_calibration(quad, (210, 297), 3.0, (1920, 1080))
    h = np.array(cal["homography"])
    pts = cv2.perspectiveTransform(quad.reshape(1, 4, 2), h)[0]
    top, bottom = np.linalg.norm(pts[1] - pts[0]), np.linalg.norm(pts[2] - pts[3])
    assert abs(top - bottom) < 1.5          # a 550 px edge and a 1050 px edge in the photo both give 630 px


def test_no_calibration_leaves_image_alone():
    img, _ = tilted_sheet()
    assert fc.apply(img, {}) is img


def test_save_and_load(tmp_path):
    p = tmp_path / "c.json"
    fc.save("blue", {"px_per_mm": 5.0, "homography": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "warp_size": [10, 10]}, p)
    assert fc.load(p)["blue"]["px_per_mm"] == 5.0
