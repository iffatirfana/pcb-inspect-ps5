import re

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
    # the real second-opinion model is off in these tests; the anomaly tests below use a fake scorer
    monkeypatch.setitem(fpl.FAB["fabrics"]["blue"]["anomaly"], "enabled", False)
    monkeypatch.setitem(fpl.CFG["llm"], "enabled", True)    # advice tests need the advisor on; config.yaml may turn it off


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
    assert rec["uncalibrated"] is False
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


# ---- anomaly second opinion (REVIEW-only) ----
def anomaly_on(monkeypatch):
    monkeypatch.setitem(fpl.FAB["fabrics"]["blue"]["anomaly"], "enabled", True)


def test_anomaly_flag_on_clean_detector_raises_pass_to_review(monkeypatch):
    anomaly_on(monkeypatch)
    rec = fpl.inspect("blue", good_img(), detector=FakeDetector([]), anomaly_scorer=FakeScorer(200, 300, 200))
    assert rec["verdict"] == "REVIEW" and rec["anomaly"]["flagged"] is True
    assert rec["findings"] and all(f["type"] == "Anomaly" and f["severity"] == "Low" for f in rec["findings"])
    assert "anomaly" in rec["timings"]
    a = fpl.advise(rec, adv())
    assert a["verdict"] == "REVIEW"


def test_anomaly_cannot_exceed_review_even_for_a_huge_region(monkeypatch):
    anomaly_on(monkeypatch)
    rec = fpl.inspect("blue", good_img(), detector=FakeDetector([]), anomaly_scorer=FakeScorer(500, 10, 10))
    assert rec["verdict"] == "REVIEW"                    # a 500 px square is far above the High area limit
    assert len(rec["findings"]) <= 3


def test_anomaly_quiet_keeps_pass(monkeypatch):
    anomaly_on(monkeypatch)
    rec = fpl.inspect("blue", good_img(), detector=FakeDetector([]), anomaly_scorer=FakeScorer(0))
    assert rec["verdict"] == "PASS" and rec["anomaly"]["flagged"] is False and rec["findings"] == []


def test_anomaly_not_run_when_detector_found_a_defect(monkeypatch):
    anomaly_on(monkeypatch)
    class Boom:
        def heatmap(self, img): raise AssertionError("anomaly must not run after a detection")
    rec = fpl.inspect("blue", good_img(), detector=FakeDetector([det("Hole", 0.8, 100, 100, 140, 140)]),
                      anomaly_scorer=Boom())
    assert rec["verdict"] == "REJECT" and "anomaly" not in rec and rec["findings"][0]["type"] == "Hole"


def test_anomaly_missing_files_is_skipped_quietly(monkeypatch):
    anomaly_on(monkeypatch)
    monkeypatch.setitem(fpl._anomaly, "blue", None)
    rec = fpl.inspect("blue", good_img(), detector=FakeDetector([]))
    assert rec["verdict"] == "PASS" and "anomaly" not in rec


def test_anomaly_not_run_on_invalid_capture(monkeypatch):
    anomaly_on(monkeypatch)
    class Boom:
        def heatmap(self, img): raise AssertionError("must not score a bad photo")
    flat = np.full((540, 960, 3), 120, np.uint8)
    rec = fpl.inspect("blue", flat, detector=FakeDetector([]), anomaly_scorer=Boom())
    assert rec["verdict"] == fpl.INVALID


def test_anomaly_finding_works_with_dashboard_actions(monkeypatch):
    import fabric_actions as fa
    anomaly_on(monkeypatch)
    rec = fpl.inspect("blue", good_img(), detector=FakeDetector([]), anomaly_scorer=FakeScorer(200, 300, 200))
    act = fa.structured_action(rec["findings"], rec["verdict"], fa.load_guidance())
    assert act["items"][0]["type"] == "Anomaly" and "Operator decides" in act["items"][0]["route"]


def test_heatmap_picture_written_when_anomaly_runs(monkeypatch, tmp_path):
    import os, cv2
    anomaly_on(monkeypatch)
    for scorer in (FakeScorer(200, 300, 200), FakeScorer(0)):          # flagged and quiet
        rec = fpl.inspect("blue", good_img(), detector=FakeDetector([]), anomaly_scorer=scorer)
        assert os.path.exists(rec["heatmap_image"])
        assert cv2.imread(rec["heatmap_image"]).shape == good_img().shape
        assert rec["anomaly"]["limit"] == fpl.FAB["fabrics"]["blue"]["anomaly"]["image_fraction"]


def test_no_heatmap_picture_when_detector_found_a_defect(monkeypatch):
    anomaly_on(monkeypatch)
    rec = fpl.inspect("blue", good_img(), detector=FakeDetector([det("Hole", 0.8, 100, 100, 140, 140)]),
                      anomaly_scorer=FakeScorer(0))
    assert rec["verdict"] == "REJECT" and "heatmap_image" not in rec


# ---- advice prompt and validator (Risk / Check / Action reply) ----
GOOD_REPLY = ("Risk: A 67 mm2 hole weakens the cloth and will tear when sewn.\n"
              "Check: Look for fraying around the hole and for more holes nearby.\n"
              "Action: Set the piece aside and tag it for scrap.")
PAYLOAD = ("Fabric: blue. Verdict: REJECT.\nDefect: Hole, High, 67 mm2, top-right. "
           "Cause: Snag. Route: Cut out the section or scrap it.")
FINDING = [{"type": "Hole", "severity": "High", "area_mm2": 66.8, "zone": "top-right"}]


def adv_obj():
    return FabricAdvisor(FabricMockBackend(), fpl.FAB["sop"])


def test_validator_accepts_risk_check_action_and_keeps_model_risk():
    a = adv_obj().validate(GOOD_REPLY, "REJECT", FINDING, PAYLOAD)
    assert a and a.risk.startswith("A 67 mm2 hole") and len(a.actions) == 2 and a.verdict == "REJECT"


def test_validator_rejects_invented_numbers():
    bad = GOOD_REPLY.replace("67 mm2", "250 mm2")
    assert adv_obj().validate(bad, "REJECT", FINDING, PAYLOAD) is None


def test_validator_rejects_release_on_a_reject():
    bad = "Risk: Small problem.\nCheck: Look at the hole.\nAction: Release the piece to the customer."
    assert adv_obj().validate(bad, "REJECT", FINDING, PAYLOAD) is None


def test_payload_and_prompt_contain_the_facts_and_fit_the_context():
    import fabric_actions as fa
    from fabric_advisor import build_user_message
    pl = build_user_message(FINDING * 3, "REJECT", "blue", fa.load_guidance(), fpl.FAB["sop"]["REJECT"])
    assert "Verdict: REJECT" in pl and "Hole, High, 67 mm2, top-right" in pl and "Route: Cut out" in pl
    prompt = adv_obj().build_prompt(pl)
    assert len(prompt) < 1300          # about 300 tokens, leaves room for the reply in the model context


def test_advise_uses_model_text_when_valid():
    class Good(FabricMockBackend):
        def generate(self, prompt):          # repeats the size from the user message, as a good reply would
            n = re.findall(r"Defect: \w+, High, (\d+) mm2", prompt.split("user<|end_header_id|>")[-1])[0]
            return GOOD_REPLY.replace("67", n)
    rec = fpl.inspect("white", good_img(), FakeScorer(40))
    a = fpl.advise(rec, FabricAdvisor(Good(), fpl.FAB["sop"]))
    assert a["source"] == "llm" and a["verdict"] == "REJECT" and a["risk"].startswith("A 25 mm2 hole")


def test_llm_off_in_config_gives_no_advice_and_keeps_verdict(monkeypatch):
    monkeypatch.setitem(fpl.CFG["llm"], "enabled", False)
    rec = fpl.inspect("blue", good_img(), detector=FakeDetector([]))
    fpl.advise(rec, advisor=None)
    assert rec["advice"]["source"] == "off" and rec["advice"]["actions"] == []
    assert rec["verdict"] == "PASS"
