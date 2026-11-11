"""Real Llama backend for llm/advisor.py, using an ExecuTorch-exported .pte.

Uses the ExecuTorch C++ `llama_main` runner (built on the Pi from the
ExecuTorch v1.5.1 source, XNNPACK on). Same interface as MockBackend:
    generate(prompt) -> str

Files (download once from the private HF repo, see download_models()):
    models/llama3_2_1b_q8da4w_512.pte
    models/tokenizer.model
"""
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

from llm.advisor import Backend

STOP_MARKERS = ("<|eot_id|>", "<|end_of_text|>")
MAX_CTX = 128   # this .pte stops at 128 tokens in total (prompt + reply)


def download_models(repo="iffatirfana/pcb-inspect-ps5-models", out="models",
                    files=("llama3_2_1b_q8da4w_512.pte", "tokenizer.model")):
    """Fetch model files from the private HF repo (needs HF_TOKEN env var)."""
    from huggingface_hub import hf_hub_download
    Path(out).mkdir(exist_ok=True)
    for f in files:
        p = hf_hub_download(repo, f, local_dir=out, token=os.environ.get("HF_TOKEN"))
        print("downloaded", p)


class ExecuTorchBackend(Backend):
    def __init__(self, pte="models/llama3_2_1b_q8da4w_512.pte",
                 tokenizer="models/tokenizer.model",
                 runner=None, max_new_tokens=150, temperature=0.0, threads=4):
        self.pte, self.tokenizer = Path(pte), Path(tokenizer)
        self.runner = runner or shutil.which("llama_main") or \
            str(Path.home() / "executorch/cmake-out/examples/models/llama/llama_main")
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.threads = threads
        self.last_stats = {}
        for p in (self.pte, self.tokenizer, Path(self.runner)):
            if not Path(p).exists():
                raise FileNotFoundError(f"Missing: {p}")

    def generate(self, prompt):
        n_prompt_tokens = len(prompt) // 4  # rough; runner prints exact counts
        cmd = [self.runner,
               f"--model_path={self.pte}",
               f"--tokenizer_path={self.tokenizer}",
               f"--prompt={prompt}",
               f"--seq_len={MAX_CTX - 1}",
               f"--temperature={self.temperature}",
               f"--cpu_threads={self.threads}"]
        t0 = time.perf_counter()
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        wall = time.perf_counter() - t0
        out = res.stdout
        # runner echoes the prompt first; keep only the generated part
        if prompt.strip() and prompt.strip()[-40:] in out:
            out = out.split(prompt.strip()[-40:], 1)[1]
        out = re.sub(r"PyTorchObserver.*", "", out, flags=re.S)
        for m in STOP_MARKERS:
            out = out.split(m)[0]
        self.last_stats = {"wall_s": round(wall, 2), **self._parse_stats(res.stdout + res.stderr)}
        m = re.search(r"PyTorchObserver\s*(\{.*\})", res.stdout + res.stderr, re.S)
        if m:
            try:
                o = json.loads(m.group(1))
                self.last_stats.update({
                    "prompt_tokens": o.get("prompt_tokens"),
                    "generated_tokens": o.get("generated_tokens"),
                    "prefill_tok_per_s": o.get("prefill_token_per_sec"),
                    "decode_tok_per_s": o.get("decode_token_per_sec"),
                    "load_s": round((o["model_load_end_ms"] - o["model_load_start_ms"]) / 1000, 2)})
            except Exception:
                pass
        return out.strip()

    @staticmethod
    def _parse_stats(text):
        stats = {}
        for key, pat in {"prompt_tokens": r"Prompt Tokens:\s*(\d+)",
                         "generated_tokens": r"Generated Tokens:\s*(\d+)",
                         "ttft_s": r"Time to first generated token:\s*([\d.]+)",
                         "gen_tok_per_s": r"Generated \d+ tokens:.*?Rate:\s*([\d.]+)",
                         "load_s": r"Model Load Time:\s*([\d.]+)"}.items():
            m = re.search(pat, text, re.S)
            if m:
                stats[key] = float(m.group(1))
        return stats


if __name__ == "__main__":
    try:
        b = ExecuTorchBackend()
        print(b.generate("<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
                         "Say hello in five words.<|eot_id|><|start_header_id|>assistant"
                         "<|end_header_id|>\n\n"))
        print(b.last_stats)
    except FileNotFoundError as e:
        print("Not ready on this machine:", e)
        print("On the Pi: build llama_main, then run download_models().")
