"""Deterministic rule engine: defect facts -> severity, verdict, SOP.

Reads kb/acceptance.yaml. First matching rule wins. No AI here:
the same input always gives the same decision.
"""
import re
from pathlib import Path

import yaml

POWER_NET = re.compile(r"^(\+?\d+V\d*|\d+V\d+|VCC.*|VDD.*|VBAT.*|VIN.*|VBUS.*|VSYS.*)$", re.I)
GROUND_NET = re.compile(r"^(GND.*|.*GND|VSS.*|0V)$", re.I)

PACKAGE_CLASSES = [
    ("fine_pitch_ic", re.compile(r"QFN|QFP|TSSOP|MSOP|DFN|BGA|LGA|SSOP", re.I)),
    ("polarized", re.compile(r"LED|DIODE|SOD|SMA|SMB|SMC|CP_|TANT|ELEC", re.I)),
    ("chip_passive", re.compile(r"\b(0201|0402|0603|0805|1206|1210)\b|^[RCL]_", re.I)),
    ("connector", re.compile(r"CONN|HEADER|PINHEADER|SOCKET|TERMINAL|USB|JST", re.I)),
    ("ic", re.compile(r"SOIC|SOT|SO-|DIP|TO-", re.I)),
]


def package_class(package):
    """Map a footprint/package name to a rule package class."""
    for name, pattern in PACKAGE_CLASSES:
        if package and pattern.search(package):
            return name
    return "other"


def _is_power(net):
    return bool(POWER_NET.match(net or ""))


def _is_ground(net):
    return bool(GROUND_NET.match(net or ""))


def condition_holds(cond, defect):
    """Evaluate a rule condition against one defect."""
    nets = [n for n in defect.get("nets", []) if n]
    if cond is None:
        return True
    if cond == "power_to_ground":
        return any(_is_power(n) for n in nets) and any(_is_ground(n) for n in nets)
    if cond == "different_nets":
        return len(set(nets)) >= 2
    if cond == "same_net":
        return len(nets) >= 2 and len(set(nets)) == 1
    if cond == "near_conductor":
        return bool(defect.get("near_conductor", False))
    if cond == "copper_exposed":
        return bool(defect.get("copper_exposed", False))
    raise ValueError(f"Unknown rule condition: {cond}")


class RuleEngine:
    def __init__(self, path="kb/acceptance.yaml"):
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        self.rules = data["rules"]
        self.default = data["default"]

    def evaluate(self, defect):
        """Return a copy of the defect with rule decision fields added."""
        pclass = package_class(defect.get("package", ""))
        for rule in self.rules:
            if rule["defect"] != defect.get("type"):
                continue
            if rule["package_class"] not in ("any", pclass):
                continue
            if not condition_holds(rule.get("condition"), defect):
                continue
            return {**defect, "package_class": pclass, "rule_id": rule["id"],
                    "severity": rule["severity"], "verdict": rule["verdict"],
                    "sop_id": rule.get("sop_id"), "rule_note": rule.get("note")}
        return {**defect, "package_class": pclass, "rule_id": "DEFAULT",
                "severity": self.default["severity"], "verdict": self.default["verdict"],
                "sop_id": self.default.get("sop_id"), "rule_note": self.default.get("note")}

    def board_verdict(self, defects):
        """Worst verdict across all defects decides the board."""
        order = {"PASS": 0, "REWORK": 1, "REJECT": 2}
        return max((d["verdict"] for d in defects), key=order.get, default="PASS")


def self_test():
    engine = RuleEngine()
    tests = [
        {"type": "solder_bridge", "package": "QFN-32", "nets": ["3V3", "GND"]},
        {"type": "solder_bridge", "package": "QFN-32", "nets": ["SDA", "SCL"]},
        {"type": "solder_bridge", "package": "SOIC-8", "nets": ["GND", "GND"]},
        {"type": "tombstone", "package": "R_0603", "nets": ["VOUT", "FB"]},
        {"type": "wrong_polarity", "package": "LED_0805", "nets": []},
        {"type": "misalignment", "package": "TSSOP-20", "nets": []},
        {"type": "scratch", "package": "", "nets": [], "copper_exposed": True},
    ]
    for t in tests:
        r = engine.evaluate(t)
        print(f"{t['type']:15} {t['package']:9} {str(t['nets']):18} -> "
              f"{r['rule_id']:7} {r['severity']:8} {r['verdict']:7} {r['sop_id']}")


if __name__ == "__main__":
    self_test()
