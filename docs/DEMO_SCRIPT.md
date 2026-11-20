# Demo video script (about 4 minutes, recorded on the Raspberry Pi 5)

Record the Pi's dashboard in the browser, with a visible terminal or a photo of the Pi in the first shot, so
it is clear the run is on the target device. Use one fixed photo set (see the end).

| Time | Show | Say |
| --- | --- | --- |
| 0:00 | The Pi 5, the webcam, the cloth. Terminal: `uname -m` and `python -m ui.fabric_app` | Offline fabric inspection on a Raspberry Pi 5, CPU only. No cloud, no accelerator. |
| 0:20 | Dashboard home | The operator uses a browser. Everything runs on this Pi. |
| 0:30 | Live camera card, Start live view | The live camera shows the cloth so the operator can place it. |
| 0:45 | Capture and select area, then Inspect full photo (normal cloth) | A normal piece. The detector finds nothing and the second-opinion model agrees, so the verdict is PASS. |
| 1:15 | Capture a piece with a hole, Crop an area around it, Inspect this part | The operator can crop one part. The detector finds a hole, measures it in millimetres and the rules give REJECT. |
| 1:50 | Result page: why, numbered picture, recommended action, Llama actions | The reason comes from the rules. The Llama 3.2 1B model, running through ExecuTorch on the Pi, words the operator's actions. |
| 2:20 | Timings line | Photo to verdict takes under half a second. The advice takes about two more seconds on the CPU. |
| 2:35 | Switch fabric to pattern, Pattern check page, add reference photos, check a photo with a missing motif | For printed cloth, a missing motif is flagged for review. It only raises PASS to REVIEW. |
| 3:15 | Recent inspections, click an entry, report page | Every inspection is stored with a report of each defect. |
| 3:40 | Operator confirmation buttons | The operator confirms or corrects the result. |
| 3:50 | Terminal: `vcgencmd measure_temp`, `get_throttled` | The Pi stays cool and unthrottled during use. |
| 4:00 | End | All stages run on the Pi. Models, code and the report are on GitHub. |

Only say numbers that are in `results/bench/summary.md` or the report.

## Photo set to prepare before recording

- Normal blue cloth: 1 photo
- Blue cloth with a hole: 1 photo
- Blue cloth with a thread defect: 1 photo (shows the REVIEW-only second opinion)
- Printed cloth, clean: `data/photos/pat_test_good_01.png`
- Printed cloth with a missing motif: `data/synth_test/pat_test_good_01_miss1.png`
- 8 or more clean printed-cloth reference photos for the Pattern check page (the clean `pat_good_*` photos)

Record at 1080p, show the mouse, and keep each action to a few seconds. Do a dry run first.
