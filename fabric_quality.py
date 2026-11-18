"""Cloth quality score: one 0-100 number and an A-D grade for the piece being inspected.

Pure rules, no model. Each finding costs points by severity, larger defects cost a little more,
and the verdict sets a ceiling so the grade never contradicts the PASS/REVIEW/REWORK/REJECT call.
"""
SEVERITY_COST = {"Low": 6.0, "Medium": 16.0, "High": 38.0}
CEILING = {"PASS": 100, "REVIEW": 84, "REWORK": 69, "REJECT": 44}
ANOMALY_COST = 12.0          # photo flagged by the second-opinion model only


def grade_for(score):
    if score >= 90:
        return "A"
    if score >= 75:
        return "B"
    if score >= 55:
        return "C"
    return "D"


def cloth_quality(findings, verdict, anomaly=None):
    """Returns {"score": int 0-100, "grade": "A".."D", "label": str}."""
    cost = 0.0
    for f in findings or []:
        base = SEVERITY_COST.get(f.get("severity"), 6.0)
        area = float(f.get("area_mm2") or 0.0)
        cost += base * (1.0 + min(area, 400.0) / 800.0)      # up to +50% for large areas
    if anomaly and anomaly.get("flagged") and not findings:
        cost += ANOMALY_COST
    score = int(round(max(0.0, min(100.0, 100.0 - cost))))
    score = min(score, CEILING.get(verdict, 100))
    grade = grade_for(score)
    label = {"A": "Excellent", "B": "Good", "C": "Fair", "D": "Poor"}[grade]
    return {"score": score, "grade": grade, "label": label}
