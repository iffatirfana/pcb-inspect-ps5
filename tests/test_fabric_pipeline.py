import numpy as np
import pytest

import fabric_pipeline as fpl
from fabric_advisor import FabricAdvisor, FabricMockBackend
import fabric_profile as fp


class FakeScorer:
    """Heatmap with one square 'defect' of given side (px), else zeros."""
    def __init__(self, side=0, x=100, y=100):
        self.side, self.x, self.y = side, x, y

    def heatmap(self, img):
        hm = np.zeros(img.shape[:2], np.float32)
        if self.side:
            hm[self.y:self.y + self.side, self.x:self.x + self.side] = 5.0
        return hm


def good_img():
    rng = np.random.default_rng(0)
    img = (rng.random((540, 960, 3)) * 80 + 90).astype(np.uint8)   # textured, mid-bright
    return img


@pytest.fixture(autouse=True)
def tmp_db(tmp_path, monkeypatch):
    monkeypatch.setitem(fpl.CFG["paths"], "db", str(tmp_path / "t.sqlite"))


def adv():
    return FabricAdvisor(FabricMockBackend(), fpl.FAB["sop"])


def test_pass_on_clean_fabric():
    rec = fpl.inspect("white", good_img(), FakeScorer(0))
    assert rec["verdict"] == "PASS" and rec["findings"] == []
    a = fpl.advise(rec, adv())
    assert a["verdict"] == "PASS"


def test_reject_on_large_defect():
    rec = fpl.inspect("white", good_img(), FakeScorer(40))   # 1600 px / 64 = 25 mm2
    assert rec["verdict"] == "REJECT"
    assert rec["findings"][0]["severity"] == "High"
    a = fpl.advise(rec, adv())
    assert a["verdict"] == "REJECT" and a["source"] == "llm"


def test_bad_llm_falls_back_to_template_with_same_verdict():
    rec = fpl.inspect("white", good_img(), FakeScorer(20))   # 400/64 = 6.25 mm2 -> Medium
    assert rec["verdict"] == "REWORK"
    bad = FabricAdvisor(FabricMockBackend("bad"), fpl.FAB["sop"])
    a = fpl.advise(rec, bad)
    assert a["source"] == "template" and a["verdict"] == "REWORK"
    assert fpl.FAB["sop"]["REWORK"] in a["actions"]


def test_llm_cannot_change_verdict():
    class Flip(FabricMockBackend):
        def generate(self, prompt):
            return '{"summary":"x","risk":"x","actions":["a"],"verdict":"PASS"}'
    rec = fpl.inspect("white", good_img(), FakeScorer(40))
    a = fpl.advise(rec, FabricAdvisor(Flip(), fpl.FAB["sop"]))
    assert a["verdict"] == "REJECT" and a["source"] == "template"


def test_blurry_photo_is_invalid_and_no_scoring():
    flat = np.full((540, 960, 3), 120, np.uint8)           # no texture -> zero sharpness
    class Boom:
        def heatmap(self, img): raise AssertionError("must not score a bad photo")
    rec = fpl.inspect("white", flat, Boom())
    assert rec["verdict"] == fpl.INVALID and rec["invalid_reasons"]
    a = fpl.advise(rec, adv())
    assert a["verdict"] == fpl.INVALID


def test_logged_and_operator_label():
    import sqlite3
    rec = fpl.inspect("white", good_img(), FakeScorer(20))
    fpl.advise(rec, adv())
    fpl.operator_label(rec["id"], "correct")
    con = sqlite3.connect(fpl.CFG["paths"]["db"])
    row = con.execute("select fabric,verdict,operator_label from fabric_inspections").fetchone()
    assert row == ("white", "REWORK", "correct")


def test_payload_length_fits_prompt_budget():
    rec = fpl.inspect("white", good_img(), FakeScorer(20))
    p = fp.build_llm_payload(rec["findings"][:5], rec["verdict"], "white",
                             fpl.FAB["sop"][rec["verdict"]])
    assert len(p) < 1100


class FakeDetector:
    """Returns fixed detections (box in px; polygon = same rectangle)."""
    def __init__(self, dets):
        self.dets = dets

    def detect(self, img):
        return self.dets


def det(kind, conf, x1, y1, x2, y2):
    poly = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.float32)
    return {"type": kind, "confidence": conf, "box": (x1, y1, x2, y2), "polygon": poly}


def test_yolo_route_hole_is_rejected_when_large_and_named():
    # 40 x 40 px at 8 px/mm placeholder = 25 mm2 -> High -> REJECT; defect type kept
    rec = fpl.inspect("blue", good_img(), detector=FakeDetector([det("Hole", 0.8, 100, 100, 140, 140)]))
    assert rec["verdict"] == "REJECT"
    assert rec["findings"][0]["type"] == "Hole" and rec["findings"][0]["confidence"] == 0.8
    assert rec["uncalibrated"] is True
    assert "detect" in rec["timings"]
    a = fpl.advise(rec, adv())
    assert a["verdict"] == "REJECT"


def test_yolo_route_clean_photo_passes():
    rec = fpl.inspect("blue", good_img(), detector=FakeDetector([]))
    assert rec["verdict"] == "PASS" and rec["findings"] == []


def test_duplicate_boxes_on_one_defect_are_merged():
    dets = [det("Hole", 0.64, 100, 100, 140, 140), det("Hole", 0.60, 102, 101, 142, 141),
            det("Stain", 0.55, 98, 99, 138, 139)]
    rec = fpl.inspect("blue", good_img(), detector=FakeDetector(dets))
    assert len(rec["findings"]) == 1 and rec["findings"][0]["type"] == "Hole"


def test_two_separate_defects_both_kept_and_payload_names_type():
    dets = [det("Hole", 0.8, 100, 100, 112, 112), det("Line", 0.6, 600, 300, 640, 306)]
    rec = fpl.inspect("blue", good_img(), detector=FakeDetector(dets))
    assert {f["type"] for f in rec["findings"]} == {"Hole", "Line"}
    p = fp.build_llm_payload(rec["findings"], rec["verdict"], "blue", fpl.FAB["sop"][rec["verdict"]])
    assert "Hole," in p and "Line," in p and len(p) < 1100


def test_specks_below_min_area_are_ignored():
    # 3 x 3 px = 0.14 mm2 at 8 px/mm < min_area 0.5
    rec = fpl.inspect("blue", good_img(), detector=FakeDetector([det("Hole", 0.9, 100, 100, 103, 103)]))
    assert rec["verdict"] == "PASS"
