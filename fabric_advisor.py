"""Llama advice for fabric. The verdict is fixed by fabric_profile rules; the model only
words the operator advice (a risk sentence, a check and an action). The reply is validated;
on failure retry once, then use the SOP line."""
import json
import re

from llm.advisor import Advice, Backend

# One worked example does more for a 1B model than any instruction. Total prompt is kept to
# about 200 tokens so that prompt plus reply fits the model's context (llm_backend MAX_CTX).
SYSTEM = (
    "You advise a textile inspector. The verdict is already decided; never change it. "
    "Use only the facts given and do not invent numbers. "
    "Reply with exactly 3 lines: Risk (why it matters), Check (what to inspect by eye), "
    "Action (next step that matches the verdict).\n"
    "Example\n"
    "Fabric: blue denim. Verdict: REJECT. Defect: Hole, High, 40 mm2, top-left. "
    "Route: cut out the section or scrap it.\n"
    "Risk: A 40 mm2 hole weakens the cloth and will tear when sewn.\n"
    "Check: Look for fraying around the hole and for more holes nearby.\n"
    "Action: Set the piece aside and cut around the hole only if enough good cloth remains."
)

_LABEL = re.compile(r"^(risk|check|action)\s*[:\-]\s*", re.I)
_NUM = re.compile(r"\d+(?:\.\d+)?")


class FabricMockBackend(Backend):
    """Laptop testing without a model. mode='bad' returns junk."""
    def __init__(self, mode="good"):
        self.mode = mode

    def generate(self, prompt):
        if self.mode == "bad":
            return "Looks fine, ship it."
        v = re.search(r"Verdict: (\w+)", prompt).group(1)
        return json.dumps({"summary": f"Verdict {v}.", "risk": "See findings.",
                           "actions": ["Follow the SOP step for this verdict."], "verdict": v})


def build_user_message(findings, verdict, fabric, guidance, sop_line):
    """Facts for the model, taken from the rules and kb/fabric_guidance.yaml (never invented).
    Keeps 'Verdict: X' in the text because the verdict must be repeated, not changed."""
    lines = [f"Fabric: {fabric}. Verdict: {verdict}."]
    for f in findings[:2]:
        g = (guidance or {}).get("defects", {}).get(f.get("type"), (guidance or {}).get("default", {}))
        route = (g.get("route") or {}).get(f["severity"], "")
        lines.append(f"Defect: {f.get('type', 'defect')}, {f['severity']}, {f['area_mm2']:.0f} mm2, "
                     f"{f['zone']}. Cause: {g.get('likely_cause', 'unknown')} Route: {route}")
    if len(findings) > 2:
        lines.append(f"+{len(findings) - 2} more defects")
    if sop_line:
        lines.append(f"SOP: {sop_line}")
    return "\n".join(lines)


class FabricAdvisor:
    def __init__(self, backend, sop):
        self.backend, self.sop = backend, sop

    def build_prompt(self, payload):
        return ("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
                f"{SYSTEM}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
                f"{payload[:600]}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")

    def validate(self, raw, verdict, findings=None, payload=None):
        """JSON reply (mock / larger models) or Risk/Check/Action lines (Llama 1B).
        Summary and verdict always come from the rules. With a payload, any number the model
        writes must appear in it, so the model cannot invent sizes."""
        raw = raw.strip()
        if "{" in raw:
            m = re.search(r"\{.*\}", raw, re.S)
            if not m:
                return None
            try:
                adv = Advice(**json.loads(m.group(0)))
            except Exception:
                return None
            if adv.verdict != verdict:
                return None
            if verdict != "PASS" and not adv.actions:
                return None
            return adv
        risk, lines = None, []
        for ln in raw.splitlines():
            ln = re.sub(r"^\s*(?:\d+[.)]|[-*•])\s*", "", ln).strip().strip('"')
            if len(ln) < 8 or ln.endswith(":"):
                continue
            m = _LABEL.match(ln)
            if m and m.group(1).lower() == "risk":
                if risk is None:
                    risk = ln[m.end():].strip()[:200]
                continue
            lines.append(ln[:200])
        lines = lines[:3]
        if not lines:
            return None
        text = " ".join(lines + ([risk] if risk else [])).lower()
        negated = re.search(r"\b(not|n't|never|until|before|hold|no)\b", text)
        if verdict != "PASS" and re.search(r"\b(release|ship|approve|accept|pass)\b", text) and not negated:
            return None
        if verdict == "PASS" and re.search(r"\b(reject|rework|scrap|set the piece aside)\b", text):
            return None
        if payload is not None:
            allowed = set(_NUM.findall(payload))
            if any(n not in allowed for n in _NUM.findall(text)):
                return None
        base = self.template(findings or [], verdict)
        return Advice(summary=base.summary, risk=risk or base.risk, actions=lines, verdict=verdict)

    def template(self, findings, verdict):
        if verdict == "PASS":
            return Advice(summary="No defects found.", risk="None.",
                          actions=[self.sop["PASS"]], verdict=verdict)
        parts = [f"{f['severity']} {f.get('type', 'defect')} {f['area_mm2']} mm2 at {f['zone']}" for f in findings]
        return Advice(summary="; ".join(parts) + ".", risk="See SOP.",
                      actions=[self.sop[verdict], self.sop["generic"]], verdict=verdict)

    def advise(self, payload, findings, verdict):
        """Returns (Advice, source) with source 'llm' | 'llm_retry' | 'template'."""
        if verdict == "PASS":      # nothing to explain: fixed SOP line, no model call
            return self.template(findings, verdict), "template"
        prompt = self.build_prompt(payload)
        for src in ("llm", "llm_retry"):
            adv = self.validate(self.backend.generate(prompt), verdict, findings, payload)
            if adv:
                return adv, src
        return self.template(findings, verdict), "template"
