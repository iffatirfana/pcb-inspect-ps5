# pcb-inspect-ps5

Offline PCB defect inspection assistant on Raspberry Pi 5, with Llama advice through ExecuTorch.
Bharat AI-SoC Challenge 2026–27, Problem Statement 5 (real-time industrial defect detection and quality-control assistant).

## What it does

A board in a fixed camera rig is photographed, aligned to a known-good ("golden") board, and checked for defects. Each defect is mapped to the exact component, pins and nets from the board's design files. A deterministic rule engine decides severity and PASS / REWORK / REJECT. Llama 3.2 1B (4-bit, ExecuTorch, XNNPACK on the Arm CPU) turns that decision and the matching repair procedure into operator instructions. The advice is validated, and a template answer built from the procedure is used if validation fails. Everything runs on the device with no network.

```
capture → preprocess → align to golden → anomaly + classify → map to component/pins/nets
        → rule engine (verdict) → Llama advice (validated JSON) → operator UI → SQLite log
```

Vision runs on the Raspberry Pi CPU (ONNX Runtime) or on a Hailo NPU (HailoRT). Llama always runs on the Arm CPU through ExecuTorch. These are separate pipelines.

## Repository layout

| Path | Stage |
|---|---|
| `capture/camera.py` | Picamera2 capture (image-file fallback off the Pi) |
| `preprocess/transforms.py` | Shared preprocessing used in training and inference |
| `align/register.py` | ORB + RANSAC homography alignment to the golden image |
| `vision/` | Anomaly detection and defect classification |
| `mapping/board_map.py` | Pixel box → board mm → reference designator, pins, nets |
| `rules/engine.py` | Deterministic acceptance rules (`kb/acceptance.yaml`) |
| `llm/advisor.py` | Prompt building, JSON validation, retry, SOP fallback |
| `llm/executorch_backend.py` | Runs the exported `.pte` with the ExecuTorch Llama runner |
| `ui/app.py` | Offline operator web page |
| `kb/` | Defect definitions, acceptance rules, repair procedures, board files |
| `tools/kicad_to_csv.py` | Generates `pos.csv` and `pads.csv` from a `.kicad_pcb` |
| `experiments/benchmark.py` | Per-stage latency, memory and temperature to CSV |
| `tests/` | Unit tests (run automatically on every push) |

## Run on a laptop (sample board)

```bash
python -m venv .venv
.venv/bin/pip install numpy opencv-python-headless pyyaml pydantic flask psutil pytest   # Windows: .venv\Scripts\pip
.venv/bin/python pipeline.py            # one inspection, printed as JSON
.venv/bin/python -m ui.app              # web UI at http://127.0.0.1:5000
.venv/bin/python -m pytest -q           # tests
.venv/bin/python -m experiments.benchmark --runs 20 --tag laptop
```

## Llama model

Llama 3.2 1B Instruct exported with ExecuTorch 1.5.1 (PyTorch 2.14.0): `llama_q8da4w` config (8-bit dynamic activations, 4-bit grouped weights, group size 128, 4-bit embeddings), KV cache with SDPA, XNNPACK backend, 512-token context. Size: 2.4 GB checkpoint → about 0.77 GB `.pte`. Model files are stored outside this repository.

## Status

| Stage | Status |
|---|---|
| Alignment, board mapping, rule engine, advisor validation, UI, logging | Implemented and tested |
| Llama export to ExecuTorch | Done |
| Llama on Raspberry Pi 5, Pi benchmarks | Pending hardware |
| Vision models (anomaly + classifier) | In progress |

## License

MIT
