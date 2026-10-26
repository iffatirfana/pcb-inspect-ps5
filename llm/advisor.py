"""Llama advisory stage: rule-engine results -> operator advice (validated JSON).

Llama never decides PASS/REWORK/REJECT - the rule engine does. Llama only
explains the decision and turns the matching SOP into operator instructions.
Every reply is validated; on failure we retry once, then use a template
answer built directly from the SOP, so the operator always gets safe advice.

The model call is pluggable (Backend.generate):
  MockBackend      - laptop testing, no model needed
  (Pi) ExecuTorch  - added later, same interface
"""
import json
import re
from pathlib import Path
from typing import List, Literal

import yaml
from pydantic import BaseModel, ValidationError, field_validator

MAX_PROMPT_CHARS = 1300   # ~330 tokens; leaves room for ~150 output tokens in 512 ctx


class Advice(BaseModel):
    summary: str
    risk: str
    actions: List[str]
    verdict: Literal["PASS", "REVIEW", "REWORK", "REJECT", "INVALID_CAPTURE"]

    @field_validator("actions")
    @classmethod
    def short_actions(cls, v):
        return [a.strip() for a in v if a and a.strip()][:8]


class Backend:
    def generate(self, prompt: str) -> str:
        raise NotImplementedError


class MockBackend(Backend):
    """Pretends to be Llama. mode='good' returns valid JSON, 'bad' returns junk."""

    def __init__(self, mode="good"):
        self.mode = mode

    def generate(self, prompt):
        if self.mode == "bad":
            return "Sure! The board looks fine, just ship it."
        verdict = re.search(r"BOARD VERDICT: (\w+)", prompt).group(1)
        refs = re.findall(r"- (\w+) ", prompt.split("DEFECTS:")[1])
        return json.dumps({
            "summary": f"Defects found on {', '.join(dict.fromkeys(refs))}.",
            "risk": "See defect list.",
            "actions": ["Do not power the board.", "Follow the listed SOP steps."],
            "verdict": verdict})


class Advisor:
    def __init__(self, backend: Backend, kb_dir="kb"):
        kb = Path(kb_dir)
        self.backend = backend
        self.sops = yaml.safe_load((kb / "sop.yaml").read_text(encoding="utf-8")) or {}
        self.defs = yaml.safe_load((kb / "defects.yaml").read_text(encoding="utf-8")) or {}

    # ---------- prompt ----------
    def build_prompt(self, defects, board_verdict):
        lines = [f"BOARD VERDICT: {board_verdict}", "DEFECTS:"]
        for d in defects:
            nets = "/".join(d.get("nets", [])) or "-"
            lines.append(f"- {d['refdes']} {d['type']} pins {d.get('pins')} nets {nets} "
                         f"severity {d['severity']} verdict {d['verdict']}")
            risk = self.defs.get(d["type"], {}).get("risk")
            if risk:
                lines.append(f"  risk: {risk}")
            if d.get("rule_note"):
                lines.append(f"  note: {d['rule_note']}")
        sop_ids = list(dict.fromkeys(d["sop_id"] for d in defects if d.get("sop_id")))
        for sid in sop_ids:
            sop = self.sops.get(sid, {})
            lines.append(f"SOP {sid}: " + " ".join(sop.get("steps", [])))
        facts = "\n".join(lines)[:MAX_PROMPT_CHARS]

        system = ("You are a PCB quality-control assistant. Use ONLY the facts given. "
                  "Do not change the verdict. In summary, name each defect type with its "
                  "component, pins and nets, e.g. \"Solder bridge on U3 pins 4-5 shorting "
                  "3V3 to GND\". Reply with JSON only: "
                  '{"summary": str, "risk": str, "actions": [str], "verdict": str}')
        return ("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
                f"{system}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
                f"{facts}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")

    # ---------- validation ----------
    @staticmethod
    def _extract_json(text):
        m = re.search(r"\{.*\}", text, re.S)
        return json.loads(m.group(0)) if m else None

    def validate(self, raw, defects, board_verdict):
        """Return (Advice, None) if valid, else (None, reason)."""
        try:
            data = self._extract_json(raw)
            if data is None:
                return None, "no JSON in reply"
            adv = Advice(**data)
        except (json.JSONDecodeError, ValidationError, TypeError) as e:
            return None, f"schema error: {e.__class__.__name__}"
        if adv.verdict != board_verdict:
            return None, f"verdict changed ({adv.verdict} != {board_verdict})"
        known = {d["refdes"] for d in defects}
        text = " ".join([adv.summary, adv.risk] + adv.actions)
        mentioned = set(re.findall(r"\b[A-Z]{1,3}\d{1,4}\b", text))
        invented = {m for m in mentioned if not re.fullmatch(r"\d?V\d*", m)} - known
        if invented:
            return None, f"unknown components mentioned: {sorted(invented)}"
        if board_verdict != "PASS" and not adv.actions:
            return None, "no actions for a failing board"
        return adv, None

    # ---------- summary ----------
    @staticmethod
    def template_summary(defects):
        parts = []
        for d in defects:
            s = f"{d['type'].replace('_', ' ').capitalize()} on {d['refdes']}"
            pins = d.get("pins") or []
            if pins:
                s += " pin" + ("s " if len(pins) > 1 else " ") + "-".join(str(p) for p in pins)
            nets = d.get("nets") or []
            if len(nets) > 1:
                s += f" shorting {nets[0]} to {nets[1]}" if d["type"] == "solder_bridge" \
                    else f" (nets {'/'.join(nets)})"
            elif nets:
                s += f" (net {nets[0]})"
            parts.append(f"{s} [{d['severity']}]")
        return "; ".join(parts) + "."

    @staticmethod
    def summary_ok(summary, defects):
        """Summary must name every defective component (weak 'Rework' summaries fail)."""
        found = set(re.findall(r"\b[A-Z]{1,3}\d{1,4}\b", summary))
        return {d["refdes"] for d in defects if d.get("refdes")} <= found

    # ---------- fallback ----------
    def template(self, defects, board_verdict):
        if not defects:
            return Advice(summary="No defects found.", risk="None.", actions=[],
                          verdict=board_verdict)
        summary = self.template_summary(defects)
        risks = [self.defs.get(d["type"], {}).get("risk", "") for d in defects]
        actions = []
        for d in defects:
            if d.get("rule_note"):
                actions.append(f"{d['refdes']}: {d['rule_note']}")
            for step in self.sops.get(d.get("sop_id"), {}).get("steps", []):
                actions.append(f"{d['refdes']}: {step}")
        return Advice(summary=summary, risk=" ".join(r for r in risks if r) or "See SOP.",
                      actions=actions or ["Send to manual inspection."],
                      verdict=board_verdict)

    # ---------- main entry ----------
    def advise(self, defects, board_verdict):
        """Returns (Advice, source): 'llm', 'llm_retry', '<that>+summary' when the
        model's actions were kept but its summary was replaced, or 'template'."""
        if not defects:
            return self.template(defects, board_verdict), "template"
        prompt = self.build_prompt(defects, board_verdict)
        for attempt in ("llm", "llm_retry"):
            raw = self.backend.generate(prompt)
            adv, _reason = self.validate(raw, defects, board_verdict)
            if adv:
                if not self.summary_ok(adv.summary, defects):
                    adv.summary = self.template_summary(defects)
                    attempt += "+summary"
                return adv, attempt
        return self.template(defects, board_verdict), "template"


def self_test():
    defects = [
        {"type": "solder_bridge", "refdes": "U3", "pins": [4, 5], "nets": ["3V3", "GND"],
         "severity": "CRITICAL", "verdict": "REWORK", "sop_id": "SOP-BRIDGE-01",
         "rule_note": "Do not power the board until the bridge is removed and verified."},
        {"type": "tombstone", "refdes": "R7", "pins": [1, 2], "nets": ["VOUT", "FB"],
         "severity": "MAJOR", "verdict": "REWORK", "sop_id": "SOP-REFLOW-01",
         "rule_note": None},
    ]
    for mode in ("good", "bad"):
        adv, src = Advisor(MockBackend(mode)).advise(defects, "REWORK")
        print(f"[mock={mode}] source={src} verdict={adv.verdict}")
        print("  summary:", adv.summary)
        print("  actions:", adv.actions[:3], "...")
    a = Advisor(MockBackend())
    p = a.build_prompt(defects, "REWORK")
    print("prompt chars:", len(p), "(~", len(p) // 4, "tokens)")
    bad = '{"summary":"Fix U9","risk":"x","actions":["a"],"verdict":"REWORK"}'
    print("hallucination check:", a.validate(bad, defects, "REWORK")[1])
    flip = '{"summary":"U3","risk":"x","actions":["a"],"verdict":"PASS"}'
    print("verdict-change check:", a.validate(flip, defects, "REWORK")[1])


if __name__ == "__main__":
    self_test()
