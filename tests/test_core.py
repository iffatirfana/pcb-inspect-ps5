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
    return BoardMap("kb/board_sample", scale_transform(0.05))


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
