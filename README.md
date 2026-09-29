# Paxini Gen3 accuracy test bench

A GUI that walks you through tests R0–R3 and RM1–RM4 from [test-plan-v2.md](test-plan-v2.md), and automatically records, films, analyzes, and reports every session.
The earlier S1–S9 and M1–M7 procedures (fixed load steps and calibration weights) are still in the code if you need them.

## What changed in v2

| | Earlier (S/M) | Current plan (R/RM) |
|---|---|---|
| Reference quantity | Fz | **Magnitude \|F\| = √(Fx²+Fy²+Fz²)**, the only quantity on a curved surface that does not depend on the sensor's frame |
| Loading | Hit a target value, hold 5 s | **Free sweep.** Press roughly however you like; the analysis classifies the events and aggregates them per bin |
| Ending a segment | Fixed number of steps | Press [Space] once the live **coverage** checklist is full |
| Multi-sensor | 8 sensors | **4 per hand**, one hand at a time |

## Install and run

| Platform | Install (once) | Run |
|---|---|---|
| Windows | `setup_windows.bat` | `run_windows.bat` (simulator: `run_windows.bat --sim`) |
| macOS/Linux | `uv venv --python 3.11 .venv && uv pip install -r requirements.txt` | `.venv/bin/python run.py --sim` |

- `--sim` runs a simulated gauge, PXSR and camera. Use it to rehearse the procedure and inspect the outputs without hardware. A simulated operator performs the sweeps for you, including surface curvature, alignment error and friction.
- `--hw` uses the real hardware. Omit both flags to follow `mode` in `config.yaml`.

## Check before connecting hardware (config.yaml)

| Item | Key | Current assumption | How to confirm |
|---|---|---|---|
| Sensor rating | `sensor_types.*.rated_N` | A 50 N, B 30 N | Manufacturer spec |
| Hand layout | `sensors[].hand` | L = A1, A2, B1, B2 / R = the rest | Your actual per-hand build (this is the unit for RM tests) |
| Gauge port and protocol | `gauge.port`, `mode`, `poll_command`, `line_regex` | COM3, Imada-compatible | Check the reply string in a serial terminal |
| Gauge accuracy | `gauge.accuracy_N` | ±2.5 N | Loads below four times this value are reported as **not judged** |
| PXSR folder and columns | `pxsr.output_dir`, `columns` | `C:/PXSR/data`, `Fx..Tz` | CSV header |
| Channel to sensor | `sensors[].channel` | A1=0 … B4=7 | Verify with RM2 |
| Contact sites | `procedure.v2.positions`, `directions` | Apex plus ±x and ±y | Keep only the faces the sensor body actually has |

## Running a session (v2)

| Step | What you do |
|---|---|
| 1 | Pick the sensors on the left (one for R tests, the **[L hand] / [R hand]** button for RM tests), then pick the test |
| 2 | **▶ Start session**, follow the prompt to zero PXSR and start its recording, then press **Space** |
| 3 | The no-load recording and the three sync taps advance on their own timer |
| 4 | Free sweep segment: put the tip on the contact point, tilt until the **\|F\| ÷ gauge ratio is at its minimum**, then press **Space** |
| 5 | Perform the ramps, pulses, holds and steps listed in the prompt, in any order. Press **Space** once **every coverage item turns green** |
| 6 | Stop the PXSR recording at the end, press **Space**, and the analysis runs automatically |

| Key | Action |
|---|---|
| Space | Confirm / end the segment |
| 1–9 | Redo the n-th free segment (when you want to re-measure a position or direction) |
| R | Back one step / redo the measurement |
| M | Marker (note) |
| Esc | Abort (the data is still saved and analyzed) |

### Telling whether you pressed well

| Tile | What it means |
|---|---|
| **\|F\| ÷ gauge** | The closer to 1, the better the tip points along the surface normal. Above 1.05, suspect a side force from friction or misalignment |
| **Force direction (from z)** | How far the force the sensor actually receives is tilted. Near 0° at the apex, near 90° on the 90° fixture |
| **Difference (\|F\| − gauge)** | The direct measure of sensor error |
| Coverage | Counts of ramps, pulses, holds and steps, plus how full each 10 % bin is. Pausing a few seconds inside an empty bin fills it |

## Outputs (sessions/&lt;time&gt;_&lt;test&gt;_&lt;sensor&gt;/)

| File | Contents |
|---|---|
| `report.html` | Verdict table, metrics, coverage, alignment diagnostics, per-bin aggregates, plots |
| `video.mp4` + `video_frames.csv` | Overlaid video (segment, event counts, ratio, direction angle) |
| `gauge.csv` / `pxsr_raw/` | Raw gauge readings, original PXSR files |
| `events.csv` | Segment and recording boundaries, re-measurements, markers |
| `metrics.csv` / `checks.csv` / `step_stats.csv` | Metrics, verdicts, per-step statistics |
| `plots/*.png` | Time series, sync, scatter and residuals, bin bars, hysteresis loops, creep and recovery, step response, direction polar |

**Results tab:** open the report, folder or video; re-analyze (re-judge after editing the criteria); attach PXSR CSVs by hand; build the **sensor comparison report**.

## What the analysis does for you

1. Corrects the PXSR clock against the gauge by cross-correlating the sync taps, using the magnitude
2. Splits contacts wherever the gauge exceeds max(0.5 N, 5σ) for at least 0.3 s
3. Classifies each contact as ramp, pulse, hold, step or other
4. Keeps only quasi-static samples, where \|dG/dt\| < 15 % F.S./s, and aggregates them into 10 % F.S. bins
5. Derives slope, nonlinearity, hysteresis and repeatability from the bins; creep from the holds; zero recovery from the releases; dynamic response from the steps
6. Diagnoses alignment from \|F\|/gauge and direction stability from the unit vector

Every threshold lives in `procedure.v2.detect` and `criteria` in `config.yaml`. Change a value and press **Re-analyze** in the results tab to judge the session again.

## Layout

| Path | Role |
|---|---|
| `paxtest/procedures.py` | Test definitions as step lists. `free` is an open-ended sweep segment |
| `paxtest/session.py` | Session recorder, step state machine, live coverage |
| `paxtest/devices/` | Gauge (serial), PXSR folder watcher and parser, camera, simulator (including surface geometry) |
| `paxtest/analysis/contacts.py` | Contact splitting, classification and coverage, shared with the live GUI |
| `paxtest/analysis/bins.py` | Gauge-to-PXSR merge, quasi-static aggregation, regression, direction, alignment |
| `paxtest/analysis/metrics_v2.py` | v2 metrics and verdicts. `metrics.py` holds the earlier S/M ones |
| `paxtest/gui/` | PySide6 GUI |
| `tools/headless_run.py` | Full pipeline check against the simulator with no GUI (`python tools/headless_run.py R1 --quick 0.1`) |
