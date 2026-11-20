# Raspberry Pi 5 run checklist

One sitting, about 90 minutes. Run everything from the repository root with the virtual environment active.
Keep the Pi cool and powered by its own USB-C supply (do not run from a laptop port).

## 1. Update and check

```
cd ~/pcb-inspect-ps5
git pull
source .venv/bin/activate
python -m pytest -q --ignore=tests/test_fabric_scorer.py
ls /dev/video*            # the webcam should be listed
vcgencmd get_throttled    # should print throttled=0x0 before you start
```

## 2. Measurements (results go to `results/bench/summary.md`)

```
python tools/pi_benchmark.py latency --n 30
python tools/pi_benchmark.py threads --threads 1,2,3,4 --n 10
python tools/pi_benchmark.py parallel --n 10
python tools/pi_benchmark.py soak --minutes 30
```

Add `--llm` to `latency` and `soak` once, to include the Llama advice (needs the runner set up as in the README).
Use `--photos <folder>` for another photo set. `all` runs the four commands in one go.

## 3. Dashboard on the Pi

```
python -m ui.fabric_app
```

Open `http://<pi-address>:5000` from a laptop. Check: live camera, capture and crop, an upload, a log report.

## 4. Collect and send back

- `results/bench/summary.md` and the CSV files in `results/bench/`
- `vcgencmd measure_temp`, `vcgencmd measure_clock arm`, `vcgencmd get_throttled` right after the soak run
- `free -h` and `uname -a`, `python --version`
- Screenshots of the dashboard on the Pi (result page, log report, live camera)
- The screen recording for the demo (see `docs/DEMO_SCRIPT.md`)

## 5. Read the results

- `get_throttled` of `0x0` all through the soak run means no under-voltage or thermal limiting.
- The thread table shows the best ONNX thread count. Set it for the final run with `FABRIC_ORT_THREADS=<n>`.
- If "both at the same time" is not faster than "one after the other", keep the sequential design.
