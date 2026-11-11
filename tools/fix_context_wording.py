"""One-off: make README.md and config.yaml say what was measured on the Pi.
The .pte file name and README said 512 tokens, but prompt + reply stops at about 127 tokens
(prompts of 162 tokens fail), so the effective context is 128. Safe to run twice."""
from pathlib import Path

EDITS = {
    "README.md": [("512-token context.",
                   "exported with a 512-token setting, but measured on the Pi the effective context is 128 tokens "
                   "(prompt plus reply stop at about 127; prompts of 162 tokens fail), so the advice prompt is kept to about 80 tokens.")],
    "config.yaml": [("context_length: 512",
                     "context_length: 128   # measured on the Pi: prompt + reply stop at ~127 tokens")],
}
for name, pairs in EDITS.items():
    p = Path(name)
    s = p.read_text(encoding="utf-8")
    for old, new in pairs:
        if old in s:
            s = s.replace(old, new)
            print(name, ": updated")
        else:
            print(name, ": nothing to change")
    p.write_text(s, encoding="utf-8")
