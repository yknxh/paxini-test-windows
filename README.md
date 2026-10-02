# Paxini Gen3 accuracy test bench

A GUI that walks you through tests R0, R1, R2, R4 and RM1–RM5 (R3 is dropped for now; RM5 presses a plate resting on one hand's 4 sensors and compares their summed force with the gauge) from [test-plan-v2.md](test-plan-v2.md), and automatically records, films, analyzes, and reports every session.
The earlier S1–S9 and M1–M7 procedures (fixed load steps and calibration weights) are still in the code if you need them.

## What changed in v2

| | Earlier (S/M) | Current plan (R/RM) |
|---|---|---|
| Reference quantity | Fz | **Magnitude \|F\| = √(Fx²+Fy²+Fz²)**, the only quantity on a curved surface that does not depend on the sensor's frame |
| Loading | Hit a target value exactly, confirm each step | **Press block** with a fixed routine: press to roughly the prompted force, **hold 3 s, release, rest 3 s**. The gauge decides when a hold and a rest are done |
| Error unit | % F.S. | **N** (types A and B have different ratings, so % F.S. is not comparable between them) |
| Multi-sensor | 8 sensors | **4 per hand**, one hand at a time |

## What you get

| Question | Where it comes from | Graph |
|---|---|---|
| How large is each sensor's error at different forces? | Every press is one point: mean \|F\| − mean gauge over the stable hold (N) | `error_*.png`, and in the comparison report all 8 sensors by type plus an error distribution per sensor |
| Does the sensor return to zero after release? | The 3 s rest after each press, sensor data only | `recovery_*.png` |
| How fast does the sensor respond? | The fast-input block (taps and quick releases), sensor data only | `response_*.png` |

Averaging over a stable hold means the time offset between the gauge (about 10 Hz) and PXSR barely affects the error. Response time is **not** measured against the gauge, because the gauge is about 20 times slower than the sensor and the sync step would absorb any delay.

## Install and run

| Platform | Install (once) | Run |
|---|---|---|
| Windows | `setup_windows.bat` | `run_windows.bat` (simulator: `run_windows.bat --sim`) |
| macOS/Linux | `uv venv --python 3.11 .venv && uv pip install -r requirements.txt` | `.venv/bin/python run.py --sim` |

- `--sim` runs a simulated gauge (10 Hz, like the real one), PXSR and camera. Use it to rehearse the procedure and inspect the outputs without hardware. A simulated operator follows the press routine for you, including surface curvature, alignment error and friction.
- `--hw` uses the real hardware. Omit both flags to follow `mode` in `config.yaml`.

## Check before connecting hardware (config.yaml)

| Item | Key | Current assumption | How to confirm |
|---|---|---|---|
| Sensor rating | `sensor_types.*.rated_N` | 25 N for both types (confirmed) | Prompted press forces are percentages of this |
| Hand layout | `sensors[].hand` | L = A1, A2, B1, B2 / R = the rest | Your actual per-hand build (this is the unit for RM tests) |
| Gauge port and protocol | `gauge.port`, `baudrate`, `mode`, `line_terminator`, `record_regex` | COM7, 2400 baud, continuous stream of fixed-width `-004.9` records (N) with no separator, ~10 Hz, `invert: true` | Check the stream with `tools/serial_sniff.py` |
| PXSR folder and columns | `pxsr.output_dir`, `columns` | `C:/PXSR/data`, `Fx..Tz` | CSV header |
| Channel to sensor | `sensors[].channel` | A1=0 … B4=7 | Verify with RM2 |
| Contact sites | `procedure.v2.positions`, `directions` | R2: ±x only; the reference is the latest R1 apex (R3 dropped) | Keep only the faces the sensor body actually has |
| Press routine | `procedure.v2.press` | Hold 3 s, rest 3 s, no-load windows 15 s (RM4 stays 30 min); every test uses the same levels 3/6/9/12 N (max 48 % of 25 N); R1 × 3, R2 sites and RM × 2, R4 × 1 | Pilot run |
| Pass criteria | `criteria` | Error ≤ 1.0 N, repeat spread ≤ 0.3 N, residual ≤ 0.1 N within 3 s (draft) | Manufacturer spec |

## Running a session (v2)

| Step | What you do |
|---|---|
| 1 | Pick the sensors on the left (one for R tests, the **[L hand] / [R hand]** button for RM tests), then pick the test |
| 2 | **▶ Start session**, follow the prompt to zero PXSR and start its recording, then press **Space** |
| 3 | The no-load recording and the three sync taps advance on their own timer |
| 4 | Press block: put the tip on the contact point, tilt until the **\|F\| ÷ gauge ratio is at its minimum**, lift off, then press **Space** |
| 5 | Follow the large prompt: **Press → about N N**, then **Hold x / 3 s** counts up while the gauge is steady, then **Release**, then **Rest x / 3 s**. The block moves on by itself after the last press |
| 6 | R1 only: the fast-input block. Tap the apex sharply with a hard object 5 times, then press to about half and release at once 5 times |
| 7 | Stop the PXSR recording at the end, press **Space**, and the analysis runs automatically |

| Key | Action |
|---|---|
| Space | Confirm / start the next step. **Ignored during a press block**, so a habitual tap does not cut it short |
| R | Redo the current step from the start (during a press block: restart the block) |
| M | Marker (note) |
| Esc | Abort (the data is still saved and analyzed) |

The main button changes to **End block early** during a press block. It asks before skipping the remaining presses.

### Telling whether you pressed well

| Tile or panel | What it means |
|---|---|
| **Press prompt** | What to do now, and the hold or rest timer. The hold timer restarts if the gauge moves more than 3 % (at least 0.3 N). The rest timer restarts if you touch the sensor |
| **\|F\| ÷ gauge** | The closer to 1, the better the tip points along the surface normal. Above 1.05, suspect a side force from friction or misalignment |
| **Force direction (from z)** | How far the force the sensor actually receives is tilted. Near 0° at the apex, near 90° on the 90° fixture |
| **Difference (\|F\| − gauge)** | The direct measure of sensor error |

## Outputs (sessions/&lt;time&gt;_&lt;test&gt;_&lt;sensor&gt;/)

| File | Contents |
|---|---|
| `report.html` | Verdict table, metrics, per-level error table, result plots first, then diagnostic plots |
| `presses.csv` | One row per press: prompted level, gauge mean, sensor \|F\| and Fx/Fy/Fz means, error (N), hold window, rest length |
| `video.mp4` + `video_frames.csv` | Overlaid video (block, press count, state, ratio, direction angle) |
| `gauge.csv` / `pxsr_raw/` | Raw gauge readings, original PXSR files |
| `events.csv` | Step boundaries, re-measurements, markers |
| `metrics.csv` / `checks.csv` / `step_stats.csv` | Metrics, verdicts, per-step statistics |
| `plots/*.png` | Results: `error_*`, `recovery_*`, `response_*` (plus `channel_map`, `rate` for RM tests). Diagnostics: `timeseries`, `sync`, `zero` |

**Results tab:** open the report, folder or video; re-analyze (re-judge after editing the criteria); attach PXSR CSVs by hand; build the **sensor comparison report**. The comparison report opens with the error of all 8 sensors against force (one panel per type), the error distribution per sensor, and zero recovery and response time side by side.

## What the analysis does for you

1. Corrects the PXSR clock against the gauge by cross-correlating the sync taps (whole-session correlation as a fallback), using the magnitude
2. Finds each press in a block: the longest stretch where the gauge stays within 3 % (at least 0.3 N) for at least the hold time
3. Averages gauge and sensor \|F\| over that stretch, minus the first 0.5 s, and records the error in N. Each press is attached to the nearest prompted level
4. Error metrics: mean error, max \|error\|, and the spread between repeats at the same level. Slope and intercept are kept as reference values for calibration
5. Zero recovery: after each release, the distance of the sensor force vector from its value before the press. Using the vector instead of \|F\| avoids counting noise as a residual
6. Response time: 10→90 % rise of the taps and 90→10 % fall of the quick releases, from the sensor signal alone
7. R2/R3 and RM2/RM3 compare the error with a reference at the same force: the apex (this session or the latest R1) or the same sensor's latest single-connection R1
8. R4 (session repeatability) reads each of the sensor's R1/R4 sessions at the same forces and reports the spread between sessions in N

Every threshold lives in `procedure.v2.press` and `criteria` in `config.yaml`. Change a value and press **Re-analyze** in the results tab to judge the session again.

## Layout

| Path | Role |
|---|---|
| `paxtest/procedures.py` | Test definitions as step lists. `press` is a press block |
| `paxtest/session.py` | Session recorder, step state machine, live press tracking |
| `paxtest/devices/` | Gauge (serial), PXSR folder watcher and parser, camera, simulator (including surface geometry) |
| `paxtest/analysis/presses.py` | Press detection (live and offline), zero recovery, rise/fall edges |
| `paxtest/analysis/metrics_v2.py` | v2 metrics, verdicts and result plots. `metrics.py` holds the earlier S/M ones |
| `paxtest/analysis/pipeline.py` | Session analysis, session list, sensor comparison report |
| `paxtest/gui/` | PySide6 GUI |
| `tools/headless_run.py` | Full pipeline check against the simulator with no GUI (`python tools/headless_run.py R1 --quick 0.3`) |
