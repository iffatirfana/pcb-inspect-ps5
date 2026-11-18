"""Evidence-grounded Llama fabric inspection advisor."""

import re

from llm.advisor import Advice, Backend


SYSTEM = (
    "You advise a textile factory operator. The verdict and severity are final; never change them. "
    "Use only the numbers given and do not invent defects or causes. "
    "Reply with exactly 3 lines, each under 200 characters:\n"
    "Risk: <effect of the defect on the cloth>\n"
    "Check: <what to inspect around it>\n"
    "Action: <what to do, consistent with the SOP>\n"
    "Example reply:\n"
    "Risk: The stain may show after dyeing and lower the grade.\n"
    "Check: Look for more stains nearby.\n"
    "Action: Mark the piece and send it for review."
)

_NUM = re.compile(r"\d+(?:\.\d+)?")
_LABELS = {
    "risk": re.compile(r"^\s*Risk\s*:\s*(.+)$", re.I),
    "check": re.compile(r"^\s*Check\s*:\s*(.+)$", re.I),
    "action": re.compile(r"^\s*Action\s*:\s*(.+)$", re.I),
}


class FabricMockBackend(Backend):
    """Mock backend for testing without ExecuTorch."""

    def __init__(self, mode="good"):
        self.mode = mode

    def generate(self, prompt):
        if self.mode == "bad":
            return "Looks fine, ship it."

        return (
            "Risk: The detected defect may affect fabric quality.\n"
            "Check: Inspect the defect region and surrounding fabric.\n"
            "Action: Follow the supplied inspection procedure."
        )


def build_user_message(findings, verdict, fabric, guidance, sop_line):
    """Build an evidence-based inspection prompt."""

    lines = [
        f"Fabric: {fabric}.",
        f"Verified verdict: {verdict}."
    ]

    for f in findings[:2]:      # the ctx512 model has room for two defects
        g = (guidance or {}).get(
            "defects", {}
        ).get(
            f.get("type"),
            (guidance or {}).get("default", {})
        )

        route = (g.get("route") or {}).get(f["severity"], "")

        lines.append(
            f"Defect: {f.get('type', 'defect')}. "
            f"Severity: {f['severity']}. "
            f"Area: {f['area_mm2']:.2f} mm2. "
            f"Location: {f['zone']}. "
            f"Possible cause: {g.get('likely_cause', 'unknown')}. "
            f"Guidance: {route}."
        )

    if len(findings) > 2:
        lines.append(f"Additional defects: {len(findings) - 2}.")

    if sop_line:
        lines.append(f"Mandatory SOP: {sop_line}")

    return "\n".join(lines)


class FabricAdvisor:

    def __init__(self, backend, sop):
        self.backend = backend
        self.sop = sop

    def build_prompt(self, payload, feedback=None):
        prompt = (
            "<|begin_of_text|>"
            "<|start_header_id|>system<|end_header_id|>\n\n"
            f"{SYSTEM}<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            f"{payload[:1800]}"
        )

        if feedback:
            prompt += f"\n\nCorrection required: {feedback}"

        return (
            prompt
            + "<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )

    def template(self, findings, verdict):
        """Deterministic fallback preserving the exact SOP."""

        if verdict == "PASS":
            return Advice(
                summary="No defects found.",
                risk="No defect-related risk identified.",
                actions=[self.sop["PASS"]],
                verdict=verdict
            )

        parts = [
            f"{f['severity']} {f.get('type', 'defect')} "
            f"{f['area_mm2']:.2f} mm2 at {f['zone']}"
            for f in findings
        ]

        return Advice(
            summary="; ".join(parts) + ".",
            risk="See verified inspection findings and SOP.",
            actions=[
                self.sop[verdict],
                self.sop["generic"]
            ],
            verdict=verdict
        )

    def validate(self, raw, verdict, findings=None, payload=None):
        """Validate the three advisory sections and preserve mandatory SOP."""

        if not raw:
            return None

        sections = {}

        for line in raw.strip().splitlines():
            line = line.strip().strip('"').strip()

            for name, pattern in _LABELS.items():
                match = pattern.match(line)
                if match and name not in sections:
                    sections[name] = match.group(1).strip()

        if set(sections) != {"risk", "check", "action"}:
            return None

        if any(len(value) < 8 for value in sections.values()):
            return None

        generated_text = " ".join(sections.values()).lower()

        # Reject recommendations contradicting the verified verdict.
        if verdict != "PASS" and re.search(
            r"\b(release|ship|approve|accept|pass)\b",
            generated_text
        ):
            return None

        if verdict == "PASS" and re.search(
            r"\b(reject|rework|scrap)\b",
            generated_text
        ):
            return None

        # Reject unsupported numerical claims.
        if payload is not None:
            allowed = set(_NUM.findall(payload))
            generated_numbers = _NUM.findall(generated_text)

            if any(number not in allowed for number in generated_numbers):
                return None

        base = self.template(findings or [], verdict)

        # The original SOP instruction is always retained.
        actions = [
            f"Inspection check: {sections['check']}",
            f"Recommendation: {sections['action']}",
        ]

        actions.extend(base.actions)

        return Advice(
            summary=base.summary,
            risk=sections["risk"],
            actions=actions,
            verdict=verdict
        )

    def advise(self, payload, findings, verdict):
        """Return Advice and source: llm, llm_retry, or template."""

        if verdict == "PASS":
            return self.template(findings, verdict), "template"

        prompt = self.build_prompt(payload)

        # First attempt
        raw = self.backend.generate(prompt)
        print("\\n[LLM DEBUG] First raw response:")
        print(repr(raw))

        adv = self.validate(raw, verdict, findings, payload)

        if adv:
            print("[LLM DEBUG] First response accepted.")
            return adv, "llm"

        print("[LLM DEBUG] First response rejected by validator.")

        # Retry with corrective feedback
        feedback = (
            "Your previous response was incomplete or invalid. "
            "Include all three sections: Risk, Check and Action. "
            "Use only supplied facts. Preserve the verified verdict "
            "and follow the mandatory SOP."
        )

        retry_prompt = self.build_prompt(payload, feedback)
        raw = self.backend.generate(retry_prompt)

        print("\\n[LLM DEBUG] Retry raw response:")
        print(repr(raw))

        adv = self.validate(raw, verdict, findings, payload)

        if adv:
            print("[LLM DEBUG] Retry response accepted.")
            return adv, "llm_retry"

        print("[LLM DEBUG] Retry rejected. Using SOP template.")

        return self.template(findings, verdict), "template"
