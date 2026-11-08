"""Llama advice for fabric. The verdict is fixed by fabric_profile rules; the model only
words the operator actions. Reply is validated; on failure retry once, then use the SOP line."""
import json
import re

from llm.advisor import Advice, Backend

SYSTEM = ("You are an assistant for a fabric inspection station. Use ONLY the facts given. "
          "Never change the verdict. Reply with JSON only: "
          '{"summary": str, "risk": str, "actions": [str], "verdict": str}. '
          "actions: 2 or 3 short operator steps that follow the SOP and verdict. "
          "Name each defect position by its zone. Do not invent defects or numbers.")


class FabricMockBackend(Backend):
    """Laptop testing without a model. mode='bad' returns junk."""
    def __init__(self, mode="good"):
        self.mode = mode

    def generate(self, prompt):
        if self.mode == "bad":
            return "Looks fine, ship it."
        v = re.search(r"Verdict \(fixed, do not change\): (\w+)", prompt).group(1)
        return json.dumps({"summary": f"Verdict {v}.", "risk": "See findings.",
                           "actions": ["Follow the SOP step for this verdict."], "verdict": v})


class FabricAdvisor:
    def __init__(self, backend, sop):
        self.backend, self.sop = backend, sop

    def build_prompt(self, payload):
        return ("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
                f"{SYSTEM}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
                f"{payload[:1100]}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")

    def validate(self, raw, verdict):
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

    def template(self, findings, verdict):
        if verdict == "PASS":
            return Advice(summary="No defects found.", risk="None.",
                          actions=[self.sop["PASS"]], verdict=verdict)
        parts = [f"{f['severity']} defect {f['area_mm2']} mm2 at {f['zone']}" for f in findings]
        return Advice(summary="; ".join(parts) + ".", risk="See SOP.",
                      actions=[self.sop[verdict], self.sop["generic"]], verdict=verdict)

    def advise(self, payload, findings, verdict):
        """Returns (Advice, source) with source 'llm' | 'llm_retry' | 'template'."""
        prompt = self.build_prompt(payload)
        for src in ("llm", "llm_retry"):
            adv = self.validate(self.backend.generate(prompt), verdict)
            if adv:
                return adv, src
        return self.template(findings, verdict), "template"
