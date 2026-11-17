# Fabric Inspection Assistant (Raspberry Pi 5, fully offline)

Bharat AI-SoC Challenge 2026-27, Problem Statement 5: real-time industrial defect detection and quality-control assistant.
Entry: textile and fabric quality control on a Raspberry Pi 5 (CPU only), with Llama 3.2 1B running through ExecuTorch.

## What it does

One photo goes in; a verdict, a repair route and two short operator actions come out, with no cloud connection.

```
photo / webcam -> YOLOv8n-seg detector (onnxruntime) -> rule engine (severity, verdict)
              -> [only if nothing found] anomaly second opinion (ResNet18 + normal-patch bank)
              -> Llama 3.2 1B on ExecuTorch (2 operator actions) -> dashboard + SQLite log
```

- **Detector:** finds Hole, Knot, Line and Stain defects, with polygon and area.
- **Anomaly second opinion:** when the detector finds nothing, a second model compares the photo with a memory bank of normal-denim patches (ResNet18 layer2+layer3 features, nearest-neighbour distance, PatchCore-style). If more than 1.5% of the photo looks abnormal, the verdict becomes **REVIEW** with up to three regions marked "Anomaly". It can only raise PASS to REVIEW; it never causes REWORK or REJECT, and it does not run when the detector already found a defect.
- **Rule engine:** Low / Medium / High severity from area, then PASS / REVIEW / REWORK / REJECT. Only rules decide the verdict.
- **Rule-generated recommendations:** disposition, urgency, repair route, what to check and likely cause per defect (`kb/fabric_guidance.yaml`).
- **Llama advisor:** words two numbered operator actions from the verdict, findings and SOP line. Its reply is validated (it must not say release/ship/approve for a non-PASS verdict); on any failure a template built from the SOP is used. Skipped on PASS.
- **Dashboard:** local web page with a colour verdict banner, annotated photo, findings table, recommendations, timings, operator feedback and history.
- **Log:** every inspection is stored in SQLite (`fabric_inspections`).

Vision runs on the Pi CPU (onnxruntime). Llama runs on the Arm CPU through ExecuTorch. No Hailo accelerator is used.

## Measured results (Raspberry Pi 5, 16 GB, our own photos of blue denim)

| Measurement | Value |
| --- | --- |
| Detector time | 134-200 ms |
| Photo to verdict | 0.31-0.43 s |
| Total with advice, persistent Llama runner | about 2.4 s (first call 4.6 s includes model load) |
| Peak memory | Python 177.7 MB + Llama runner 773.8 MB |
| False alarms at confidence 0.5 | 0 of 71 normal photos |
| Hole recall at confidence 0.5 | 12 of 15 |
| Thread (Line) recall at confidence 0.5, detector alone | 2 of 6 (weak) |

### Anomaly second opinion (held-out photos, same denim fabric)

The memory bank was built from the 51 train photos only. The pixel threshold was chosen on the 10 validation photos only. The test normals and all defect photos were never used for building or tuning. Run through the full pipeline (laptop CPU, about 0.38 s per photo):

| Set | Result |
| --- | --- |
| Validation normals (10) | 10 PASS (threshold set here) |
| Test normals (10) | 8 PASS, 2 REVIEW (false alarms) |
| Holes (15) | detector alone 12 REJECT; with second opinion 15 of 15 flagged (3 as REVIEW) |
| Threads (6) | detector alone 2 REJECT; with second opinion 6 of 6 flagged (4 as REVIEW) |

These are small sets from one fabric and one camera setup, so they show the idea works but are not an accuracy claim. Pi timing for the anomaly step has not been measured yet.

## Limitations (read these)

- The anomaly second opinion is trained on normal blue denim only and flags about 2 in 10 held-out normal photos for a quick manual check. It is advisory: it can raise PASS to REVIEW and nothing more.
- Trained on blue denim only. A yellow patterned cloth gave false Holes; other fabrics need new data.
- The pixel-to-mm scale in `kb/fabric.yaml` is a placeholder until calibrated from a ruler photo (`calibrated: false`), so sizes and severities are estimates. The dashboard warns about this.
- The Llama file is named `..._512.pte`, but measured on the Pi the effective context is **128 tokens** (prompt plus reply stop at about 127; prompts of 162 tokens fail). The advice prompt is therefore about 80 tokens and the model writes only two short actions.
- Experiments that did not help (tiling, a larger 960x960 detector) are described in the technical report.

## Repository layout (fabric entry)

| Path | Purpose |
| --- | --- |
| `fabric_yolo.py` | onnxruntime-only YOLOv8-seg detector (letterbox, NMS, mask polygons) |
| `fabric_scorer.py` | ResNet18 feature extractor plus normal-patch bank, giving an anomaly heatmap |
| `fabric_profile.py` | detections to findings in mm, severity, verdict, Llama payload |
| `fabric_actions.py`, `kb/fabric_guidance.yaml` | rule-generated recommendation fields |
| `fabric_advisor.py` | prompt, validation, retry and template fallback for Llama |
| `fabric_pipeline.py` | capture, quality gate, detect, rules, advice, SQLite log; CLI and `--live` webcam mode |
| `ui/fabric_app.py` | offline operator dashboard |
| `llm/executorch_backend.py` | runs the `.pte` with the ExecuTorch `llama_main` runner (persistent server mode) |
| `tools/patch_llama_main_server.py` | adds `--server` mode to ExecuTorch's `llama_main` |
| `kb/fabric.yaml` | fabric settings, thresholds (including the anomaly pixel threshold and 1.5% rule), SOP text per verdict |
| `tests/` | unit tests |

The repository also contains an earlier PCB inspection profile (`pipeline.py`, `ui/app.py`, `kb/boards`); it is not part of this entry.

## Run on a laptop (no Llama needed; a mock advisor is used)

```bash
python -m venv .venv
.venv/bin/pip install numpy opencv-python-headless pyyaml pydantic onnxruntime flask psutil pytest   # Windows: .venv\Scripts\pip
.venv/bin/python -m pytest tests/test_fabric_profile.py tests/test_fabric_actions.py tests/test_fabric_app.py tests/test_fabric_pipeline.py -q
.venv/bin/python fabric_pipeline.py blue path/to/photo.jpg
.venv/bin/python -m ui.fabric_app      # http://127.0.0.1:5000
```
This needs `models/best_169.onnx` (see Models below).

## Run on the Raspberry Pi 5

1. Flash Raspberry Pi OS (64-bit), enable SSH, clone this repository, create a virtual environment and run `pip install -r requirements.txt`. On the Pi the Python `executorch` package is not needed; the C++ runner is used.
2. Put the model files in `models/` (see Models).
3. Build ExecuTorch v1.5.1 and its Llama runner following `examples/models/llama/README.md` in the ExecuTorch repository. Apply `python tools/patch_llama_main_server.py` to `examples/models/llama/main.cpp`, then rebuild with `cmake --workflow --preset llama-release` from `examples/models/llama`. The runner is expected at `~/executorch/cmake-out/examples/models/llama/llama_main`.
4. Start the dashboard: `python -m ui.fabric_app`, then open `http://<pi-ip>:5000`.
5. Webcam: a USB camera at `/dev/video0` (MJPG 1920x1080) is used by the Capture button and by `python fabric_pipeline.py blue --live`.

## Models

Model files are not stored in Git except the detector and the two anomaly files (force-added because `.gitignore` excludes `*.onnx`).

| File | What | Source |
| --- | --- | --- |
| `models/best_169.onnx` (13.7 MB) | YOLOv8n-seg detector, 640x384 input | trained by the team on the public denim dataset below |
| `models/anomaly/resnet18_fused.onnx` (11 MB) | ResNet18 layer2+layer3 feature extractor, 512x512 input | ResNet18 ImageNet weights exported to ONNX by the team |
| `models/anomaly/bank_blue_trainonly.npy` (7.7 MB) | 5000 normal-denim patch features (384 values each) | built from the 51 training photos |
| `models/llama3_2_1b_q8da4w_512.pte` (802 MB) | Llama 3.2 1B Instruct, ExecuTorch, 8da4w, XNNPACK | exported by the team with ExecuTorch 1.5.1 |
| `models/tokenizer.model` (2.2 MB) | Llama 3 tokenizer | from Llama 3.2 |

The Llama files are fetched with `llm.executorch_backend.download_models()` from the team's Hugging Face repository.

## Credits and licences

- Defect dataset: public denim defect dataset on Roboflow (workspace i-g-geethashri, project bkdn-huylv-datn-mt9s6, version 3), licence CC BY 4.0.
- Anomaly method: PatchCore-style nearest-neighbour patch scoring (Roth et al., "Towards Total Recall in Industrial Anomaly Detection", CVPR 2022); feature extractor ResNet18 with ImageNet weights (torchvision).
- YOLOv8 training used ultralytics (AGPL-3.0); inference in this repository uses onnxruntime only.
- Llama 3.2 1B: Meta, under the Llama 3.2 Community Licence. **Built with Llama.**
- ExecuTorch: PyTorch / Meta, BSD-style licence.
- Code in this repository: MIT.

## Team

Person 1 (detector, rule engine, Pi deployment, Llama stage, dashboard) and Person 2 (anomaly detector). Faculty Mentor: Aathilakshmi, Associate Professor, ECE department.
