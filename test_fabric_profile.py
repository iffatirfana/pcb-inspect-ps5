import numpy as np
import pytest
from fabric_profile import *

PPM = 8.0  # 8 px per mm


def blank(h=480, w=640):
    return np.zeros((h, w), np.float32)


def blob(hm, x, y, side_px, val=1.0):
    hm[y:y + side_px, x:x + side_px] = val


def test_ruler_scale():
    assert px_per_mm_from_ruler(400, 50) == 8.0
    with pytest.raises(ValueError):
        px_per_mm_from_ruler(0, 50)


def test_clean_heatmap_passes():
    r = inspect(blank(), 0.5, PPM)
    assert r["verdict"] == "PASS" and r["findings"] == []


def test_area_in_mm2():
    hm = blank(); blob(hm, 100, 100, 16)          # 16x16 px = 2x2 mm = 4 mm2
    f = findings_from_heatmap(hm, 0.5, PPM)
    assert len(f) == 1 and f[0]["area_mm2"] == 4.0 and f[0]["severity"] == "Medium"


def test_tiny_speck_ignored():
    hm = blank(); blob(hm, 50, 50, 2)             # 4 px = 0.06 mm2
    assert findings_from_heatmap(hm, 0.5, PPM) == []


def test_verdict_levels():
    hm = blank(); blob(hm, 10, 10, 10)            # 100 px = 1.56 mm2 -> Low
    assert inspect(hm, 0.5, PPM)["verdict"] == "REVIEW"
    hm = blank(); blob(hm, 10, 10, 16)            # 4 mm2 -> Medium
    assert inspect(hm, 0.5, PPM)["verdict"] == "REWORK"
    hm = blank(); blob(hm, 10, 10, 40)            # 25 mm2 -> High
    assert inspect(hm, 0.5, PPM)["verdict"] == "REJECT"


def test_many_low_becomes_rework():
    hm = blank()
    for i in range(5):
        blob(hm, 20 + i * 60, 20, 10)
    assert inspect(hm, 0.5, PPM)["verdict"] == "REWORK"


def test_zone_and_position():
    hm = blank(480, 640); blob(hm, 560, 400, 16)  # bottom right
    f = findings_from_heatmap(hm, 0.5, PPM)[0]
    assert f["zone"] == "bottom-right"
    assert 70 < f["x_mm"] < 72 and 50 < f["y_mm"] < 52


def test_payload_contains_fixed_verdict():
    hm = blank(); blob(hm, 10, 10, 40)
    r = inspect(hm, 0.5, PPM, fabric="white", sop_text="Mark and set aside.")
    assert "Verdict: REJECT" in r["llm_payload"]
    assert "Mark and set aside." in r["llm_payload"]


def test_crop():
    img = np.zeros((100, 200, 3), np.uint8)
    assert crop_image(img, (10, 20, 50, 40)).shape == (40, 50, 3)
    assert crop_image(img, None).shape == img.shape
