"""Unit tests for the deterministic parts of the pipeline. Run: python -m pytest -q"""
import json

import cv2
import numpy as np
import pytest

from align.register import Aligner, synthetic_board
from llm.advisor import Advisor, MockBackend
from mapping.board_map import BoardMap, scale_transform
from rules.engine import RuleEngine, package_class
from tools.kicad_to_csv import convert

BRIDGE = {"type": "solder_bridge", "refdes": "U3", "pins": [4, 5], "package": "QFN-32",
          "nets": ["3V3", "GND"]}


# ---------- rule engine ----------
@pytest.fixture(scope="module")
def engine():
    return RuleEngine("kb/acceptance.yaml")


@pytest.mark.parametrize("nets,rule,sev,verdict", [
    (["3V3", "GND"], "R01", "CRITICAL", "REWORK"),
    (["+5V", "GND"], "R01", "CRITICAL", "REWORK"),
    (["SDA", "SCL"], "R02", "MAJOR", "REWORK"),
    (["GND", "GND"], "R03", "MINOR", "PASS"),
])
def test_bridge_severity_depends_on_nets(engine, nets, rule, sev, verdict):
    r = engine.evaluate({**BRIDGE, "nets": nets})
    assert (r["rule_id"], r["severity"], r["verdict"]) == (rule, sev, verdict)


def test_package_classes():
    assert package_class("QFN-32_5x5mm_P0.5mm") == "fine_pitch_ic"
    assert package_class("R_0603_1608Metric") == "chip_passive"
    assert package_class("LED_0805_2012Metric") == "polarized"


def test_board_verdict_is_worst(engine):
    assert engine.board_verdict([{"verdict": "PASS"}, {"verdict": "REWORK"}]) == "REWORK"
    assert engine.board_verdict([]) == "PASS"


# ---------- board mapping ----------
@pytest.fixture(scope="module")
def board():
    return BoardMap("kb/boards/sample_synthetic", scale_transform(0.05))


def test_bridge_maps_to_pins_and_nets(board):
    r = board.locate([348, 292, 356, 308])
    assert r["refdes"] == "U3" and r["pins"] == [4, 5] and r["nets"] == ["3V3", "GND"]


def test_empty_area_maps_to_nothing(board):
    assert board.locate([900, 900, 920, 905])["refdes"] is None


# ---------- advisor safety ----------
@pytest.fixture(scope="module")
def defects(engine):
    return [engine.evaluate(BRIDGE)]


def test_valid_llm_reply_accepted(defects):
    adv, src = Advisor(MockBackend("good")).advise(defects, "REWORK")
    assert src == "llm" and adv.verdict == "REWORK"


def test_junk_reply_falls_back_to_sop(defects):
    adv, src = Advisor(MockBackend("bad")).advise(defects, "REWORK")
    assert src == "template" and any("flux" in a.lower() for a in adv.actions)


def test_hallucinated_component_rejected(defects):
    bad = json.dumps({"summary": "Replace U9", "risk": "x", "actions": ["a"], "verdict": "REWORK"})
    assert "U9" in Advisor(MockBackend()).validate(bad, defects, "REWORK")[1]


def test_verdict_change_rejected(defects):
    flip = json.dumps({"summary": "U3 ok", "risk": "x", "actions": [], "verdict": "PASS"})
    assert "verdict changed" in Advisor(MockBackend()).validate(flip, defects, "REWORK")[1]


def test_prompt_fits_context(defects):
    assert len(Advisor(MockBackend()).build_prompt(defects * 3, "REWORK")) // 4 < 400


# ---------- alignment ----------
def test_alignment_recovers_pose():
    g = synthetic_board()
    h, w = g.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), 2.0, 1.0)
    M[0, 2] += 12
    t = cv2.warpAffine(g, M, (w, h))
    _, H, stats = Aligner(g).align(t)
    corners = np.float32([[[100, 100]], [[700, 500]]])
    moved = cv2.transform(corners, M)
    back = cv2.perspectiveTransform(moved, H)
    assert np.abs(back - corners).max() < 2.0 and stats["inliers"] > 100


# ---------- KiCad converter ----------
def test_kicad_rotation(tmp_path):
    convert("tests/fixtures_board.kicad_pcb", tmp_path)
    pads = (tmp_path / "pads.csv").read_text().splitlines()
    assert "C1,1,110.0,60.825,+3V3" in pads and "R1,2,100.825,50.0,VOUT" in pads


# ---------- board profiles ----------
def test_profiles_listed_and_readiness():
    from mapping.profiles import list_profiles
    profiles = list_profiles("kb/boards")
    assert profiles["sample_synthetic"].ready
    test_board = profiles["ps5_testboard_v1"]
    assert not test_board.ready
    assert any("golden" in p for p in test_board.problems())


def test_profile_maps_bridge():
    from mapping.profiles import list_profiles
    bm = list_profiles("kb/boards")["sample_synthetic"].board_map
    d = bm.locate([348, 292, 356, 308])
    assert d["refdes"] == "U3" and sorted(d["nets"]) == ["3V3", "GND"]


# ---------- tiling ----------
def test_tiles_cover_image_and_stitch_back():
    from preprocess.transforms import stitch_maps, tile_image
    img = np.zeros((3040, 4056, 3), np.uint8)
    tiles = tile_image(img, 256, 32)
    covered = np.zeros(img.shape[:2], bool)
    for x, y, t in tiles:
        assert t.shape[:2] == (256, 256)
        covered[y:y + 256, x:x + 256] = True
    assert covered.all()
    maps = [(x, y, np.ones((64, 64), np.float32)) for x, y, _ in tiles]
    heat = stitch_maps(maps, img.shape[:2], 256)
    assert heat.shape == img.shape[:2] and np.allclose(heat, 1.0)


# ---------- advisor summary ----------
def test_weak_summary_replaced(defects):
    class Weak(MockBackend):
        def generate(self, prompt):
            return '{"summary": "Rework", "risk": "r", "actions": ["a"], "verdict": "REWORK"}'
    adv, src = Advisor(Weak()).advise(defects, "REWORK")
    assert src == "llm+summary" and "U3" in adv.summary and "3V3" in adv.summary


# ---------- quality gate ----------
def test_quality_gate_rejects_blur_and_dark():
    from vision.quality import check_alignment, check_capture
    img = cv2.imread("kb/boards/sample_synthetic/test_board.png")
    assert check_capture(img)[0]
    assert not check_capture(cv2.GaussianBlur(img, (0, 0), 6))[0]
    assert not check_capture((img * 0.1).astype(np.uint8))[0]
    assert not check_alignment({"inliers": 5, "reproj_err_px": 1.0})[0]


# ---------- REVIEW / confidence ----------
def test_unknown_and_low_confidence_go_to_review(engine):
    assert engine.evaluate({"type": "unknown_anomaly", "conf": 0.9})["verdict"] == "REVIEW"
    low = engine.evaluate({**BRIDGE, "nets": ["SDA", "SCL"], "conf": 0.3})
    assert low["verdict"] == "REVIEW" and low["severity"] == "MAJOR"    # severity kept
    assert engine.evaluate({"type": "short_circuit", "conf": 0.3})["verdict"] == "REJECT"
    assert engine.board_verdict([{"verdict": "REVIEW"}, {"verdict": "PASS"}]) == "REVIEW"
    assert engine.board_verdict([{"verdict": "REVIEW"}, {"verdict": "REWORK"}]) == "REWORK"


# ---------- inspectors ----------
def test_yolo_decode_and_nms():
    from vision.inspectors import decode_yolo, nms
    out = np.zeros((1, 4 + 2, 3), np.float32)
    out[0, :4, 0], out[0, 5, 0] = [100, 100, 20, 10], 0.9
    out[0, :4, 1], out[0, 5, 1] = [101, 100, 20, 10], 0.6     # duplicate of 0
    out[0, :4, 2], out[0, 4, 2] = [300, 300, 20, 10], 0.1     # below threshold
    dets = nms(decode_yolo(out, ["open_circuit", "short_circuit"], 0.25, offset=(640, 0)))
    assert len(dets) == 1 and dets[0][2] == "short_circuit" and dets[0][0][0] == 730


def test_classifier_without_model_is_unknown():
    from vision.inspectors import Classifier
    c = Classifier("models/does_not_exist.onnx", ["solder_bridge", "normal"])
    assert c.classify(np.zeros((100, 100, 3), np.uint8), [40, 40, 60, 60]) == ("unknown_anomaly", None)


def test_bare_board_routes_to_yolo(tmp_path):
    from mapping.profiles import BoardProfile
    from vision.inspectors import InspectorError, get_inspector
    (tmp_path / "profile.yaml").write_text("name: t\nboard_type: bare\n")
    p = BoardProfile(tmp_path)
    assert p.inspector == "yolo"
    with pytest.raises(InspectorError):
        get_inspector(p, {"yolo": {"model": "models/does_not_exist.onnx"}})


def test_golden_diff_finds_new_blob():
    from vision.golden_diff import GoldenDiff

    def colour_features(x):                       # stand-in for the CNN: 8x8 colour means
        n, c, h, w = x.shape
        f = x.reshape(n, c, h // 8, 8, w // 8, 8).mean((3, 5))
        return np.concatenate([f, np.ones((n, 1, h // 8, w // 8), np.float32)], 1)

    golden = cv2.imread("kb/boards/sample_synthetic/golden.png")
    gd = GoldenDiff(extractor=colour_features)
    gd.set_references([golden])
    gd.calibrate([golden.copy()], margin=1.0)
    gd.threshold = max(gd.threshold, 0.02)
    bad = golden.copy()
    cv2.circle(bad, (500, 400), 14, (230, 230, 230), -1)
    boxes = gd.detect(gd.score(bad))
    assert len(boxes) == 1
    x1, y1, x2, y2 = boxes[0]["bbox"]
    assert x1 < 500 < x2 and y1 < 400 < y2


def test_image_to_board_transform():
    from mapping.board_map import image_to_board
    assert np.allclose(image_to_board(0.05, (10, 20)), scale_transform(0.05, (10, 20)))
    T = image_to_board(0.1, (100, 100), rotation_deg=180)
    assert np.allclose(T @ [110, 100, 1], [-1, 0, 1])


# ---------- full pipeline ----------
def test_pipeline_verdict_then_advice(monkeypatch):
    import pipeline
    from capture.camera import Camera
    monkeypatch.setattr(pipeline, "CAMERA", Camera(source="file"))  # never the live camera
    rec = pipeline.inspect("sample_synthetic")
    assert rec["verdict"] == "REWORK" and rec["advice"] is None and rec["id"]
    adv = pipeline.advise(rec)
    assert adv["verdict"] == "REWORK" and "U3" in adv["summary"]
    assert {"to_verdict", "llm", "total"} <= set(rec["timings"])


def test_pipeline_invalid_capture():
    import pipeline
    assert pipeline.inspect("sample_synthetic", np.zeros((600, 800, 3), np.uint8))["verdict"] == "INVALID_CAPTURE"
    flipped = cv2.flip(cv2.imread("kb/boards/sample_synthetic/golden.png"), 1)   # "wrong board"
    rec = pipeline.inspect("sample_synthetic", flipped)
    assert rec["verdict"] == "INVALID_CAPTURE" and rec["invalid_reasons"]


def test_camera_sources_fall_back_to_file():
    from capture.camera import Camera
    c = Camera(source="file", fallback_image="kb/boards/sample_synthetic/test_board.png")
    assert c.cam is None and c.source.startswith("file:")
    assert c.capture().ndim == 3
    u = Camera(source="usb", usb_index=99,
               fallback_image="kb/boards/sample_synthetic/test_board.png")
    assert u.cam is None and "usb camera unavailable" in u.reason
    assert u.capture().ndim == 3
