import fabric_quality as fq


def f(sev, area):
    return {"severity": sev, "area_mm2": area}


def test_clean_cloth_is_grade_a():
    q = fq.cloth_quality([], "PASS")
    assert q["score"] == 100 and q["grade"] == "A"


def test_reject_never_above_grade_d():
    q = fq.cloth_quality([f("High", 50)], "REJECT")
    assert q["score"] <= 44 and q["grade"] == "D"


def test_verdict_ceiling_applies():
    assert fq.cloth_quality([f("Low", 1)], "REVIEW")["score"] <= 84
    assert fq.cloth_quality([f("Medium", 5)], "REWORK")["score"] <= 69


def test_more_defects_lower_score():
    one = fq.cloth_quality([f("Low", 5)], "PASS")["score"]
    two = fq.cloth_quality([f("Low", 5), f("Low", 5)], "PASS")["score"]
    assert two < one


def test_anomaly_only_flag_costs_points():
    q = fq.cloth_quality([], "REVIEW", {"flagged": True})
    assert q["score"] < 100


def test_score_clamped():
    q = fq.cloth_quality([f("High", 900)] * 10, "REJECT")
    assert 0 <= q["score"] <= 100
