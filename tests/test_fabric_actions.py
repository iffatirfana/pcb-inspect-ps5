import fabric_actions as fa

G = fa.load_guidance()


def f(type_, sev, conf=0.9, zone="middle-centre", area=5.0):
    return {"type": type_, "severity": sev, "confidence": conf, "zone": zone, "area_mm2": area}


def test_reject_hole_high():
    a = fa.structured_action([f("Hole", "High", area=14.2)], "REJECT", G)
    assert a["urgency"] == "Immediate"
    assert "Set aside" in a["disposition"]
    assert a["items"][0]["route"].startswith("Cut out")


def test_pass_has_no_items():
    a = fa.structured_action([], "PASS", G)
    assert a["disposition"] == "Release" and a["items"] == []


def test_unknown_type_uses_default():
    a = fa.structured_action([f("Weird", "Medium")], "REWORK", G)
    assert a["items"][0]["route"] == G["default"]["route"]["Medium"]


def test_notes():
    a = fa.structured_action([f("Line", "Low", conf=0.5)], "REVIEW", G, uncalibrated=True)
    assert len(a["notes"]) == 2


def test_every_defect_has_all_severities():
    for name, g in G["defects"].items():
        assert set(g["route"]) == {"Low", "Medium", "High"}, name
