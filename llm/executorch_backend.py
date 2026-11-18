"""Real Llama backend for llm/advisor.py, using an ExecuTorch-exported .pte.

Uses the ExecuTorch C++ `llama_main` runner (built on the Pi from the
ExecuTorch v1.5.1 source, XNNPACK on). Same interface as MockBackend:
    generate(prompt) -> str

Files (download once from the private HF repo, see download_models()):
    models/llama3_2_1b_q8da4w_512.pte
    models/tokenizer.model
"""
import atexit
import json
import os
import re
import select
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from llm.advisor import Backend

STOP_MARKERS = ("<|eot_id|>", "<|end_of_text|>")
MAX_CTX = 480   # total tokens (prompt + reply); the ctx512 .pte accepts up to 512
READY, DONE, ERR = b"<<LLM_READY>>", b"<<LLM_DONE>>", b"<<LLM_ERROR>>"


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
                 runner=None, max_new_tokens=150, temperature=0.0, threads=4,
                 persistent=True):
        self.pte, self.tokenizer = Path(pte), Path(tokenizer)
        self.runner = runner or shutil.which("llama_main") or \
            str(Path.home() / "executorch/cmake-out/examples/models/llama/llama_main")
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.threads = threads
        self.last_stats = {}
        self.persistent = persistent
        self._proc = None
        self._server_failed = False
        atexit.register(self.close)
        for p in (self.pte, self.tokenizer, Path(self.runner)):
            if not Path(p).exists():
                raise FileNotFoundError(f"Missing: {p}")

    def _base_cmd(self):
        return [self.runner,
                f"--model_path={self.pte}",
                f"--tokenizer_path={self.tokenizer}",
                f"--seq_len={MAX_CTX - 1}",
                f"--temperature={self.temperature}",
                f"--cpu_threads={self.threads}"]

    # ---- persistent server: the model is loaded once and answers many prompts ----
    def _read_until(self, markers, timeout):
        fd, buf, end = self._proc.stdout.fileno(), b"", time.time() + timeout
        while True:
            for m in markers:
                if m in buf:
                    return buf.split(m, 1)[0], m
            left = end - time.time()
            if left <= 0:
                raise TimeoutError("runner did not answer in time")
            if not select.select([fd], [], [], left)[0]:
                continue
            chunk = os.read(fd, 65536)
            if not chunk:
                raise EOFError("runner exited")
            buf += chunk

    def _start_server(self):
        self._proc = subprocess.Popen(self._base_cmd() + ["--server"], stdin=subprocess.PIPE,
                                      stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self._read_until([READY], 120)

    def close(self):
        p, self._proc = self._proc, None
        if p is not None and p.poll() is None:
            try:
                p.stdin.write(b"quit\n"); p.stdin.flush(); p.wait(timeout=3)
            except Exception:
                p.kill()

    def _generate_server(self, prompt):
        if self._proc is None or self._proc.poll() is not None:
            self._start_server()
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write(prompt)
        try:
            self._proc.stdin.write((f.name + "\n").encode()); self._proc.stdin.flush()
            raw, marker = self._read_until([DONE, ERR], 120)
        finally:
            os.unlink(f.name)
        if marker == ERR:
            raise RuntimeError("runner reported an error")
        return raw.decode("utf-8", "replace")

    # ---- one-shot: start the runner for every prompt (slower, always works) ----
    def _generate_oneshot(self, prompt):
        res = subprocess.run(self._base_cmd() + [f"--prompt={prompt}"],
                             capture_output=True, text=True, timeout=300)
        return res.stdout + "\x00" + res.stderr

    def generate(self, prompt):
        t0 = time.perf_counter()
        raw, mode = None, "oneshot"
        if self.persistent and not self._server_failed:
            for attempt in (1, 2):
                try:
                    raw, mode = self._generate_server(prompt), "server"
                    break
                except Exception:
                    self.close()
            if raw is None:
                self._server_failed = True
        if raw is None:
            raw = self._generate_oneshot(prompt)
        wall = time.perf_counter() - t0
        out, _, err = raw.partition("\x00")
        # runner echoes the prompt first; keep only the generated part
        if prompt.strip() and prompt.strip()[-40:] in out:
            out = out.split(prompt.strip()[-40:], 1)[1]
        out = re.sub(r"PyTorchObserver.*", "", out, flags=re.S)
        for m in STOP_MARKERS:
            out = out.split(m)[0]
        self.last_stats = {"wall_s": round(wall, 2), "mode": mode, **self._parse_stats(raw)}
        m = re.search(r"PyTorchObserver\s*(\{.*?\})", raw, re.S)
        if m:
            try:
                o = json.loads(m.group(1))
                self.last_stats.update({
                    "prompt_tokens": o.get("prompt_tokens"),
                    "generated_tokens": o.get("generated_tokens"),
                    "prefill_tok_per_s": o.get("prefill_token_per_sec"),
                    "decode_tok_per_s": o.get("decode_token_per_sec")})
                if mode == "oneshot":
                    self.last_stats["load_s"] = round(
                        (o["model_load_end_ms"] - o["model_load_start_ms"]) / 1000, 2)
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
