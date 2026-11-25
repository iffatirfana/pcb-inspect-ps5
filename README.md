# Fabric Inspection Assistant (Raspberry Pi 5, fully offline)

Bharat AI-SoC Challenge 2026-27, Problem Statement 5: real-time industrial defect detection and quality-control assistant.
Entry: textile and fabric quality control on a Raspberry Pi 5 (CPU only), with Llama 3.2 1B running through ExecuTorch.

## What it does

One photo goes in; a verdict, a repair route and a short three-line operator note come out, with no cloud connection.

```
photo / webcam -> YOLOv8n-seg detector (onnxruntime) -> rule engine (severity, verdict)
              -> [only if nothing found] anomaly second opinion (ResNet18 + normal-patch bank)
              -> Llama 3.2 1B on ExecuTorch (Risk / Check / Action note) -> dashboard + SQLite log
```

- **Detector:** finds Hole, Knot, Line and Stain defects, with polygon and area.
- **Anomaly second opinion:** when the detector finds nothing, a second model compares the photo with a memory bank of normal-denim patches (ResNet18 layer2+layer3 features, nearest-neighbour distance, PatchCore-style). If more than 1.5% of the photo looks abnormal, the verdict becomes **REVIEW** with up to three regions marked "Anomaly". It can only raise PASS to REVIEW; it never causes REWORK or REJECT, and it does not run when the detector already found a defect. The dashboard shows an **anomaly heatmap** card (per-pixel score, flag threshold outlined) for these photos.
- **Rule engine:** Low / Medium / High severity from area, then PASS / REVIEW / REWORK / REJECT. Only rules decide the verdict.
- **Rule-generated recommendations:** disposition, urgency, repair route, what to check and likely cause per defect (`kb/fabric_guidance.yaml`).
- **Llama advisor:** writes three lines (Risk, Check, Action) from the verdict, findings and SOP line, using a ~300-token chat-template prompt with one worked example. Its reply is validated: it must not say release/ship/approve for a non-PASS verdict, and every number must come from the facts it was given. On any failure, after one retry, a template built from the SOP is used. Skipped on PASS.
- **Dashboard:** local web page with a colour verdict banner, annotated photo, findings table, recommendations, the Llama note, the anomaly heatmap when the second opinion fires, timings, operator feedback and history.
- **Log:** every inspection is stored in SQLite (`fabric_inspections`).

Vision runs on the Pi CPU (onnxruntime). Llama runs on the Arm CPU through ExecuTorch. No Hailo accelerator is used.

## Measured results (Raspberry Pi 5, 16 GB, our own photos of blue denim)

| Measurement | Value |
| --- | --- |
| Detector time | 583 ms (about 1.7 FPS, 960 px input) |
| Photo to verdict | 0.99 s median |
| Llama advice, persistent runner | 7 to 14 s per reply (16 tokens per second) |
| Peak memory | Python 226 MB + Llama runner 731 MB (0.96 GB of 16 GB) |
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

These are small sets from one fabric and one camera setup, so they show the idea works but are not an accuracy claim. On the Pi the anomaly step took about 0.76 s on a clean photo (full clean-photo run 1.11 s). Full pipeline, fresh process: hole photo REJECT with model-written advice in 5.09 s including model load; two consecutive requests to the persistent runner ran without reload.

## Pattern cloth: missing-motif check

`fabric_pattern.py` looks for a missing printed motif on repeating-print cloth (fabric key `pattern`). It finds the
print grid in one photo and flags a position that has a motif on both sides but almost no printed ink.
It needs no reference photo and never gives more than REVIEW.

| Test | Result |
|---|---|
| 30 synthetic missing motifs (made from normal photos) | 27 found at the right place |
| 14 clean photos not used to build the rule | 1 flagged (1 false gap) |

Limits: the missing motifs are synthetic and the clean photos show only a few cloth positions, so treat the
numbers as an estimate. Wrong-type or flipped motifs are not detected (the hand-printed motifs vary too much).
It works only on a regular repeating print. A reference-photo comparison was tried first and rejected: it
flagged 6 of 14 clean photos because the cloth layout differs between areas. A DINOv2 Siamese model from the
team was also tested; alone it flags too many regions on clean cloth, so it is not in the pipeline.

## Limitations

- The anomaly second opinion is trained on normal blue denim only and flags about 2 in 10 held-out normal photos for a quick manual check. It is advisory: it can raise PASS to REVIEW and nothing more.
- Trained on blue denim only. A yellow patterned cloth gave false Holes, and on grey cloth and in shadows the anomaly second opinion flags regions (REVIEW only); other fabrics need new data or a bank built from clean photos of that fabric.
- The pixel-to-mm scale in `kb/fabric.yaml` (`px_per_mm`) comes from a ruler photo and is valid only for that camera, height and image size; redo it if the setup changes. If `calibrated` is false the dashboard warns that sizes and severities are estimates.
- The older `..._512.pte` file measured only about 127 tokens of usable context on the Pi. The rebuilt `llama3_2_1b_q8da4w_ctx512.pte` was tested on the Pi with prompts of 7, 213 and 469 tokens (469 + 42 generated = 511). `config.yaml` caps prompt plus reply at 480 tokens; the advice prompt is about 300 tokens.
- Experiments that did not help (tiling, a larger 960x960 detector) are described in the technical report.

## Repository layout (fabric entry)

| Path | Purpose |
| --- | --- |
| `fabric_yolo.py` | onnxruntime-only YOLOv8-seg detector (letterbox, NMS, mask polygons) |
| `fabric_scorer.py` | ResNet18 feature extractor plus normal-patch bank, giving the anomaly heatmap (also drawn on the dashboard) |
| `fabric_profile.py` | detections to findings in mm, severity, verdict, Llama payload |
| `fabric_actions.py`, `kb/fabric_guidance.yaml` | rule-generated recommendation fields |
| `fabric_advisor.py` | prompt, validation, retry and template fallback for Llama |
| `fabric_pipeline.py` | capture, quality gate, detect, rules, advice, SQLite log; CLI and `--live` webcam mode |
| `ui/fabric_app.py` | offline operator dashboard |
| `llm/executorch_backend.py` | runs the `.pte` with the ExecuTorch `llama_main` runner (persistent server mode) |
| `tools/patch_llama_main_server.py` | adds `--server` mode to ExecuTorch's `llama_main` |
| `kb/fabric.yaml` | fabric settings, thresholds (including the anomaly pixel threshold and 1.5% rule), SOP text per verdict |
| `tests/` | unit tests |

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
| `models/llama3_2_1b_q8da4w_ctx512.pte` (766 MB) | Llama 3.2 1B Instruct, ExecuTorch, 8da4w, XNNPACK | exported by the team with ExecuTorch 1.5.1 |
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

Iffat Irfana M, I.G Geethashri and Harini V (detector, rule engine, Pi deployment, Llama stage, dashboard, anomaly detector). Faculty Mentor: Aathilakshmi, Associate Professor, ECE department.
