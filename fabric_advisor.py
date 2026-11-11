"""Llama advice for fabric. The verdict is fixed by fabric_profile rules; the model only
words the operator actions. Reply is validated; on failure retry once, then use the SOP line."""
import json
import re

from llm.advisor import Advice, Backend

SYSTEM = "Operator assistant. Reply with 2 short numbered actions only."


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


class FabricAdvisor:
    def __init__(self, backend, sop):
        self.backend, self.sop = backend, sop

    def build_prompt(self, payload):
        # The .pte has a 128-token context (prompt + reply), so keep the prompt short.
        return ("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
                f"{SYSTEM}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
                f"{payload[:400]}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")

    def validate(self, raw, verdict, findings=None):
        """JSON reply (mock / larger models) or numbered action lines (128-token model).
        Summary, risk and verdict always come from the rules; the model words the actions."""
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
        lines = []
        for ln in raw.splitlines():
            ln = re.sub(r"^\s*(?:\d+[.)]|[-*\u2022])\s*", "", ln).strip().strip('"')
            if len(ln) >= 8 and not ln.endswith(":"):
                lines.append(ln[:140])
        lines = lines[:3]
        if not lines:
            return None
        text = " ".join(lines).lower()
        negated = re.search(r"\b(not|n't|never|until|before|hold|no)\b", text)
        if verdict != "PASS" and re.search(r"\b(release|ship|approve|accept|pass)\b", text) and not negated:
            return None
        if verdict == "PASS" and re.search(r"\b(reject|rework|scrap|set the piece aside)\b", text):
            return None
        base = self.template(findings or [], verdict)
        return Advice(summary=base.summary, risk=base.risk, actions=lines, verdict=verdict)

    def template(self, findings, verdict):
        if verdict == "PASS":
            return Advice(summary="No defects found.", risk="None.",
                          actions=[self.sop["PASS"]], verdict=verdict)
        parts = [f"{f['severity']} {f.get('type', 'defect')} {f['area_mm2']} mm2 at {f['zone']}" for f in findings]
        return Advice(summary="; ".join(parts) + ".", risk="See SOP.",
                      actions=[self.sop[verdict], self.sop["generic"]], verdict=verdict)

    def advise(self, payload, findings, verdict):
        """Returns (Advice, source) with source 'llm' | 'llm_retry' | 'template'."""
        prompt = self.build_prompt(payload)
        for src in ("llm", "llm_retry"):
            adv = self.validate(self.backend.generate(prompt), verdict, findings)
            if adv:
                return adv, src
        return self.template(findings, verdict), "template"
