# pcb-inspect-ps5

Offline PCB defect inspection assistant on Raspberry Pi 5, with Llama advice through ExecuTorch.
Bharat AI-SoC Challenge 2026–27, Problem Statement 5 (real-time industrial defect detection and quality-control assistant).

## What it does

A board in a fixed camera rig is photographed. A quality gate rejects unusable photos (blur, exposure, wrong board). The photo is aligned to the board's known-good ("golden") photo and checked by the inspector for its board type:

- **Assembled boards:** golden-feature difference finds anything that differs from the golden board (no training per board), and a MobileNetV3 classifier names each finding. Unnamed findings stay "unknown anomaly" and go to human review.
- **Bare boards:** a YOLO detector finds trained copper defects (open, short, mouse bite, spur, spurious copper, pin hole).

Each defect is mapped to the exact component, pins and nets from the board's design files. A deterministic rule engine decides severity and the verdict, which is shown and logged immediately. Llama 3.2 1B (4-bit, ExecuTorch, XNNPACK on the Arm CPU) then turns that decision and the matching repair procedure into operator instructions. The advice is validated, and a template answer built from the procedure is used if validation fails. The operator marks each finding correct or wrong, and can report misses. Everything runs on the device with no network.

```
capture → quality gate → align to golden ─┬─ assembled: golden-diff → MobileNetV3 ─┬→ map to component/pins/nets
                                          └─ bare:      YOLO ──────────────────────┘
        → rule engine → VERDICT (shown + logged) → Llama advice → operator ✓/✗ → SQLite log
```

Verdicts: **PASS**, **REVIEW** (a person decides: unknown anomaly or low detector confidence), **REWORK**, **REJECT**, or **INVALID CAPTURE** (photo unusable, no verdict). Detector confidence and severity are kept separate: severity comes from the rules, confidence only decides whether a person must confirm.

Vision runs on the Raspberry Pi CPU (ONNX Runtime) or on a Hailo NPU (HailoRT). Llama always runs on the Arm CPU through ExecuTorch. These are separate pipelines.

## Repository layout

| Path | Stage |
|---|---|
| `capture/camera.py` | Picamera2 capture (image-file fallback off the Pi) |
| `preprocess/transforms.py` | Shared preprocessing used in training and inference |
| `align/register.py` | ORB + RANSAC homography alignment to the golden image |
| `vision/quality.py` | Image-quality gate (sharpness, exposure, alignment) |
| `vision/inspectors.py` | Inspector per board type: golden-diff + classifier, YOLO, demo |
| `vision/golden_diff.py` | Golden-feature difference anomaly detection (ONNX on the Pi) |
| `mapping/board_map.py` | Pixel box → board mm → reference designator, pins, nets |
| `rules/engine.py` | Deterministic acceptance rules (`kb/acceptance.yaml`) |
| `llm/advisor.py` | Prompt building, JSON validation, retry, SOP fallback |
| `llm/executorch_backend.py` | Runs the exported `.pte` with the ExecuTorch Llama runner |
| `ui/app.py` | Offline operator web page |
| `mapping/profiles.py` | Board profiles: one folder per board type in `kb/boards/` |
| `kb/` | Defect definitions, acceptance rules, repair procedures (shared by all boards) |
| `kb/boards/<board>/` | Per-board data: `profile.yaml`, `golden.png`, `pos.csv`, `pads.csv` |
| `tools/kicad_to_csv.py` | Generates `pos.csv` and `pads.csv` from a `.kicad_pcb` |
| `tools/setup_board.py` | Computes a board's golden-diff threshold from good photos |
| `experiments/eval_visa.py` | Golden-diff accuracy on the VisA PCB benchmark |
| `experiments/benchmark.py` | Per-stage latency, memory and temperature to CSV |
| `tests/` | Unit tests (run automatically on every push) |

## Board profiles

The pipeline, models and rules are shared. Each board type only adds a folder:

```
kb/boards/<board_id>/
  profile.yaml   name, board_type (assembled | bare), calibration (mm_per_px, origin_px,
                 rotation_deg, flip_y), anomaly_threshold
  golden.png     photo of a known-good board in the fixture
  refs/          ~4 more good photos (assembled boards: extra references)
  calib/         5-10 good photos used only to set the threshold
  pos.csv        component positions   (python -m tools.kicad_to_csv board.kicad_pcb kb/boards/<board_id>)
  pads.csv       pad positions and nets
```

For an assembled board, run `python -m tools.setup_board <board_id>` once after adding the photos; it writes `anomaly_threshold` into `profile.yaml`. The operator picks the board in the UI dropdown (or `python pipeline.py <board_id>`); `board:` in `config.yaml` is the default. Boards with missing files are listed as not ready, with the reason.

Models (ONNX, stored outside the repository): `models/golden_diff_resnet18.onnx` (make once with `GoldenDiff().export_onnx()`), `models/classifier.onnx` (MobileNetV3; without it every finding is an unknown anomaly), `models/yolo_bare.onnx` (bare boards).

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
| Quality gate, alignment, board-type routing, board mapping, rule engine, advisor validation, UI with verification, logging | Implemented and tested |
| Golden-diff anomaly detection | Implemented; VisA evaluation pending |
| Llama export to ExecuTorch | Done |
| Llama on Raspberry Pi 5, Pi benchmarks | Pending hardware |
| MobileNetV3 classifier, YOLO for bare boards | Training in progress |

## License

MIT
