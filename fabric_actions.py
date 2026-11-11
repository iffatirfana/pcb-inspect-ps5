"""Rule-generated recommendation fields for the dashboard.

Everything here comes from the verdict, the findings and kb/fabric_guidance.yaml.
No language model is involved, so these fields are never wrong or reworded:
the model only words the two short actions shown beside them.
"""
from pathlib import Path

import yaml

GUIDANCE_PATH = Path("kb/fabric_guidance.yaml")
LOW_CONFIDENCE = 0.6


def load_guidance(path=GUIDANCE_PATH):
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def structured_action(findings, verdict, guidance, uncalibrated=False):
    """Returns {disposition, urgency, items:[{type,severity,zone,check,likely_cause,route}], notes}."""
    va = guidance["verdict_action"].get(verdict, {"disposition": "Operator decides", "urgency": "Normal"})
    items = []
    for f in findings[:5]:
        g = guidance["defects"].get(f.get("type"), guidance["default"])
        items.append({
            "type": f.get("type", "Defect"),
            "severity": f["severity"],
            "zone": f["zone"],
            "area_mm2": f["area_mm2"],
            "check": g["check"],
            "likely_cause": g["likely_cause"],
            "route": g["route"].get(f["severity"], guidance["default"]["route"][f["severity"]]),
        })
    notes = []
    if any(f.get("confidence", 1.0) < LOW_CONFIDENCE for f in findings):
        notes.append("Low detector confidence on at least one finding: confirm by eye.")
    if uncalibrated:
        notes.append("Scale not calibrated: sizes in mm are estimates.")
    if len(findings) > 5:
        notes.append("%d more findings not listed." % (len(findings) - 5))
    return {"disposition": va["disposition"], "urgency": va["urgency"], "items": items, "notes": notes}
