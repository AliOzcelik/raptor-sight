# Detection benchmark plan

## 1. Questions the plan answers

1. **Speed:** which detector gives the lowest and most consistent delay from camera to bounding box on this Pi?
2. **Quality:** which one detects your target most reliably, with steady boxes and few dropped frames?
3. **Value of the NPU:** how much the Hailo-10H gains over the Pi 5 CPU alone, in speed and in CPU load.
4. **Input to the simulator:** the delay and noise figures that the latency and detector-noise models in the simulator will need. This makes the benchmark useful later, not a dead end.

## 2. What gets tested

| ID | Model | Runs on | Format | Hailo's published accuracy (mAP) / FPS | Role |
|---|---|---|---|---|---|
| H1 | YOLO26n | Hailo-10H | HEF | 38.6 / 233 | Main candidate |
| H2 | YOLO26s | Hailo-10H | HEF | 45.8 / 125 | Main candidate |
| H3 | YOLO11n | Hailo-10H | HEF | 38.0 / 302 | Mature alternative |
| H4 | YOLO11s | Hailo-10H | HEF | 45.5 / 142 | Mature alternative |
| H5 | DAMO-YOLO T | Hailo-10H | HEF | 42.3 / 205 | Different design family |
| H6 | DAMO-YOLO S | Hailo-10H | HEF | 45.3 / 115 | Different design family |
| H7 | YOLOv8m | Hailo-10H | HEF (already installed) | 49.2 / 76 | Sanity check. **Dropped after L0:** it matched Hailo's FPS, and YOLO11m is more accurate at the same speed |
| H8 | YOLO11m | Hailo-10H | HEF (already installed) | 50.2 / 71 | Upper end of Hailo speed vs. accuracy |
| H9 | NanoDet-RepVGG | Hailo-10H | HEF (416×416 input) | 28.6 / 837 | Speed reference only; accuracy too low for the tracker |
| H10 | YOLO26m | Hailo-10H | HEF | 51.0 / 59 | Most accurate model that may fit the delay budget (added after L0) |
| C1 | YOLO26n | Pi 5 CPU | NCNN | — | NPU vs. CPU (same model as H1) |
| C2 | YOLO26n | Pi 5 CPU | ONNX Runtime | — | Compares the two CPU runtimes |
| C3 | RF-DETR-N | Pi 5 CPU | ONNX Runtime | — | Accuracy ceiling (transformer) |
| T1 | H1 + TrackerNano | Hailo + CPU | — | — | Detect every Nth frame, track in between |

All models except NanoDet use 640×640 input. The Hailo numbers are COCO accuracy after quantization and FPS with one frame at a time, measured by Hailo on an x86 PC, so expect lower FPS on the Pi. Testing other resolutions means recompiling the Hailo models, which needs Hailo's compiler on an x86 PC, so it is phase 2 (see §8).

## 3. Measurement levels

Each level isolates one part of the pipeline, so we know where the time goes.

### L0: Chip speed alone, no Python
- **Tool:** `hailortcli benchmark` and `hailortcli run2` (on the Hailo-10H, `run2` replaces `run`). `--measure-temp` works on this board; `--measure-power` is not supported.
- **Measures:** frames per second when frames are sent one at a time vs. 8 at a time, time per frame on the chip, overall delay including the CPU side, and chip temperature.
- **Purpose:** the best-case number, checked against Hailo's published table to see how much the Pi's single PCIe lane costs.

### L1: Model on recorded frames
- **Input:** a fixed set of about 1,000 recorded camera frames, identical for every model.
- **Timing:** each stage is timed separately with `time.perf_counter_ns()`:
  - preprocessing (resize, letterbox, color conversion)
  - inference
  - post-processing (decoding the outputs, NMS where the model needs it, rescaling the boxes)
- **Run rules:** 100 warm-up frames that aren't counted, then at least 1,000 measured frames, repeated 3 times.
- **Statistics:** median, 95th and 99th percentile, worst case, and standard deviation for each stage.
- **Two modes:**
  - **One frame at a time:** the cleanest delay figure.
  - **Pipelined:** frame N+1 is prepared while frame N is on the chip. This raises frame rate, so we record how much it adds to the delay.

### L2: Live camera, full pipeline
- **Delay per frame:** `t_box_ready − SensorTimestamp`, using the timestamp Picamera2 attaches to each frame. Check once that this timestamp uses the same clock as `time.monotonic_ns()`.
- **Also records:**
  - actual frames per second
  - the gaps between consecutive results, to see how regular they are
  - dropped frames
- **Camera modes:** 1536×864 at 30, 60 and 120 fps. This shows whether a faster camera lowers the overall delay or just piles up a queue of frames.
- **LED test:** a GPIO pin switches on an LED in front of the camera at time t0. We then measure when the first frame shows it and when the pipeline reacts. This catches exposure and readout time, which the sensor timestamp misses. It also gives one ground-truth delay, which goes straight into the latency model.

### L3: Accuracy
- **Standard check:** COCO val2017 (500-image subset, then all 5,000 images), scored with `pycocotools`. This runs our own pipeline, including the 8-bit quantization and our preprocessing, which is why the scores can come out below Hailo's published numbers.
- **Tracking check on your own videos.** This matters more than COCO. We record clips with the NoIR camera:
  - Target: **no fixed class.** Anything can be a target, so the clips contain several object types (people, cups, bottles, chairs, balls, phones and so on), and accuracy is reported per class and overall.
  - Scenes: still target at 3 distances, slow movement, fast movement, partial occlusion, indoor and daylight lighting.
  - Reference labels: either labeled by hand, or produced on a PC by a large model (RF-DETR-L) and checked by hand.

| Metric | How it's measured | Why it matters for the tracker |
|---|---|---|
| Recall per frame | Fraction of frames where the target is detected | How often the policy is "blind" |
| Dropout run length | Distribution of consecutive missed frames | Sets how dropouts are simulated and how the policy is trained to handle them |
| Center jitter | Spread of the box center (in pixels) on a still target | Basis for the detector-noise model |
| Size jitter | Spread of box width and height on a still target | Noise in the `w, h` observations |
| Small-object recall | Recall grouped by box size | How far away a target can still be tracked |
| False positives per minute | Boxes of the target class where no target is | Risk of locking onto the wrong object |

### L4: Load on the Pi and stability over time
- **CPU and RAM:** `psutil`, sampled at 2 Hz.
- **Temperature and throttling:** SoC temperature (`vcgencmd measure_temp`), throttling flags (`vcgencmd get_throttled`), CPU clock, and Hailo temperature.
- **Power:** a USB-C power meter if you have one; otherwise `vcgencmd pmic_read_adc`, which reads the Pi 5's power-chip currents.
- **Soak test:** the top 2–3 candidates run for 15 minutes, checking whether delay or frame rate drifts as the board heats up.

## 4. Fixed conditions

Every run records:
- the PCIe generation and link speed
- the CPU frequency governor (fixed to `performance` for the benchmarks)
- the fan state and room temperature
- the HailoRT, model and OS versions
- the camera mode

Only one benchmark process runs at a time, with no desktop apps open. Each configuration runs 3 times, and results are reported as the median of the 3 runs.

## 5. Comparisons

| # | Comparison | Question it answers |
|---|---|---|
| 1 | H1–H6: YOLO26 vs. YOLO11 vs. DAMO-YOLO at the same size class | Which model family is best at the same size? |
| 2 | H1 vs. C1 vs. C2 (same YOLO26n) | How much does the NPU actually gain? |
| 3 | Best Hailo model vs. C3 (RF-DETR) | How much accuracy do we give up for speed? |
| 4 | H1 vs. T1 | Does detect-plus-track lower delay without losing the target? |
| 5 | Camera at 30 / 60 / 120 fps | What gives the lowest overall delay? |
| 6 | PCIe Gen 2 vs. Gen 3 | Does the PCIe speed setting matter? |
| 7 | One frame at a time vs. pipelined | How much frame rate do we gain, and how much delay does it cost? |

## 6. Output

```
perception/bench/
├── models.yaml           # candidate list (§2)
├── run_l0_hailortcli.py  # L0
├── bench_offline.py      # L1
├── bench_live.py         # L2 + L4
├── led_latency.py        # L2 LED test
├── eval_accuracy.py      # L3 (COCO + custom clips)
├── results/<date>/<run_id>/   # raw logs, per-frame timings, summary.csv
└── report.ipynb          # summary CSV + plots
```

The report contains:
- a cumulative chart of the full delay for each model
- an accuracy-vs-delay chart showing which models are best trade-offs
- a jitter histogram
- the distribution of dropout run lengths
- a sheet of simulator parameters (delay distribution, center noise in pixels, dropout probability), ready to load into `configs/`

## 7. Proposed pass criteria

These are draft numbers to adjust:
- **Frame rate:** at least 30 fps sustained over the 15-minute run.
- **Full delay:** 99th percentile at or below 40 ms, worst case at or below 60 ms.
- **Recall:** at least 95% on the "slow movement" clips, with dropout runs mostly 2 frames or shorter.
- **CPU load:** at or below 50% of one core, so there is room left for the policy and servo control.
- **Final pick:** the most accurate model on your own videos that meets all of the above.

## 8. Order of work, and open questions

1. Set up the HailoRT Python API, download the HEF files for the 10H, and run L0 (half a day). **Done for PCIe Gen 2 and Gen 3** (see below). The Python API comes from the HailoRT 5.4.0 wheel in `~/Downloads`, installed in `.venv`. The apt package `python3-h10-hailort` is version 5.1.1, which doesn't match the 5.4.0 runtime.
2. Run L1 and L2, including the LED test (1–2 days). **L1 done** (all classes). **L2 done:** the live-camera run, the ISP stream and the exposure sweep (`bench_live.py`, see below). Still to do for L2: the LED test. Target class: **person**.
3. Record and label the custom clips, then run L3.
4. Run L4 and write the report; the result is the chosen detector and the simulator parameters.

Open questions:
- **Is an x86 PC available, ideally with an NVIDIA GPU?** It's needed for Hailo's compiler, for testing other resolutions and for compiling a custom-trained model later.
- ~~Which target class should we start with?~~ **None:** anything can be a target, so every benchmark decodes all 80 COCO classes (decided 2026-10-24).
- **Does "anything" mean COCO's 80 classes, or also objects outside them?** The Hailo detectors only know COCO's 80 classes. Objects outside them would need a class-agnostic approach: a single-object tracker started from a box you select (like T1's tracker), or an open-vocabulary detector.
- **Is there a USB-C power meter and an LED with a resistor?** If not, L4 falls back to the Pi's built-in power readings, and the LED test can use the Pi's built-in activity LED if it's in view of the camera.

---

# First results

## L0: Hailo-10H chip speed, PCIe Gen 2 (2026-10-24)

**Setup:**
- PCIe: 5.0 GT/s, 1 lane (Gen 2; the hardware supports up to 8.0 GT/s)
- HailoRT and firmware: 5.4.0, HEF files from Model Zoo v5.4.0
- CPU governor: `ondemand`
- Run length: 15 s per measurement
- Throttling: none (`throttled=0x0`)

**Raw data:** `results/2026-10-24/l0_gen2/` (`summary.csv`, `environment.json`, `raw/`)

FPS was measured with one frame at a time.

| ID | Model | FPS | % of Hailo's published FPS | Time on the chip (ms) | Overall delay (ms) | Extra delay outside the chip (ms) |
|---|---|---|---|---|---|---|
| H1 | YOLO26n | 187 | 80% | 4.0 | 20.1 | 16.0 |
| H2 | YOLO26s | 117 | 94% | 7.7 | 23.1 | 15.4 |
| H3 | YOLO11n | 228 | 75% | 3.2 | **16.5** | 13.4 |
| H4 | YOLO11s | 140 | 99% | 6.9 | 20.3 | 13.4 |
| H5 | DAMO-YOLO T | 173 | 84% | 5.1 | 18.6 | 13.4 |
| H6 | DAMO-YOLO S | 111 | 96% | 8.5 | 21.3 | 12.9 |
| H7 | YOLOv8m | 76 | 100% | 12.7 | 26.1 | 13.4 |
| H8 | YOLO11m | 70 | 100% | 13.9 | 27.3 | 13.4 |
| H9 | NanoDet | 439 | 52% | 2.9 | 9.6 | 6.7 |

### What this shows

1. **Every model meets the 30 fps target easily.** Even the m-size models run at about 70 fps.
2. **Most of the delay happens outside the chip.** The chip itself takes 3–14 ms, but about 13 ms is added on top for every model: moving data over PCIe plus converting it on the CPU. YOLO26 adds about 2.5 ms more, because its output tensors are larger and haven't been decoded yet. This is the main thing to dig into in L1 (Python), and it may be reducible.
3. **Small models are held back by the PCIe link.** The m models reach 100% of Hailo's published FPS, while the n models and NanoDet reach only 52–80%. Gen 3 should help exactly these models. *(Disproved by the Gen 3 run below: their FPS didn't change.)*
4. **Sending 8 frames at a time gives no gain:** the FPS is the same as one at a time on this setup. That's fine for us, since we process one frame at a time anyway.
5. **Run order affected temperature.** The chip warmed steadily from 46°C to 67°C over the run. That didn't change these results, but the longer tests will add a cool-down pause between models.

**Current standing:** YOLO11n has the lowest overall delay among the useful models, and YOLO11s and DAMO-YOLO T are the best balance of speed and accuracy. YOLO26 loses some ground to the extra conversion work after the chip.

## L0: PCIe Gen 2 vs. Gen 3 (comparison #6, 2026-10-24)

**Setup:** same as the Gen 2 run, except PCIe is 8.0 GT/s, 1 lane (Gen 3, `dtparam=pciex1_gen=3` in `/boot/firmware/config.txt`). Throttling: none (`throttled=0x0`).

**Raw data:** `results/2026-10-24/l0_gen3/` (`summary.csv`, `environment.json`, `raw/`)

The time on the chip itself is the same in both runs (to within 0.1 ms), so all the savings come from the PCIe transfer.

| ID | Model | FPS Gen 2 → Gen 3 | % of Hailo's published FPS (Gen 3) | Overall delay Gen 2 → Gen 3 (ms) | Extra delay outside the chip Gen 2 → Gen 3 (ms) | Saved |
|---|---|---|---|---|---|---|
| H1 | YOLO26n | 187 → 186 | 80% | 20.1 → 17.1 | 16.0 → 13.0 | −3.0 ms |
| H2 | YOLO26s | 117 → 124 (+5%) | 99% | 23.1 → 20.1 | 15.4 → 12.4 | −3.0 ms |
| H3 | YOLO11n | 228 → 230 | 76% | 16.5 → **14.9** | 13.4 → 11.7 | −1.6 ms |
| H4 | YOLO11s | 140 → 142 | 100% | 20.3 → 18.6 | 13.4 → 11.8 | −1.6 ms |
| H5 | DAMO-YOLO T | 173 → 192 (+11%) | 94% | 18.6 → 15.7 | 13.4 → 10.6 | −2.9 ms |
| H6 | DAMO-YOLO S | 111 → 113 | 98% | 21.3 → 18.4 | 12.9 → 9.9 | −2.9 ms |
| H7 | YOLOv8m | 76 → 76 | 100% | 26.1 → 24.4 | 13.4 → 11.8 | −1.6 ms |
| H8 | YOLO11m | 70 → 71 | 100% | 27.3 → 25.7 | 13.4 → 11.8 | −1.6 ms |
| H9 | NanoDet | 439 → 459 (+5%) | 55% | 9.6 → 8.8 | 6.7 → 5.9 | −0.8 ms |

### What this shows

1. **Gen 3 cuts 1.6–3 ms off the overall delay for every model.** The extra delay outside the chip drops from about 13.4 ms to about 11.75 ms for YOLO11 and YOLOv8. YOLO26 and DAMO-YOLO save about twice as much, because they send larger raw output tensors back to the CPU. This agrees with the tensor-size explanation from the Gen 2 run.
2. **Bandwidth doesn't hold back the small models' FPS** (correcting point 3 above). YOLO26n and YOLO11n are flat (186 and 230 FPS), still only 75–80% of Hailo's published figures. Something else is, probably the overhead of switching between parts of the model on the chip or per-frame host overhead. Only DAMO-YOLO T (+11%) and NanoDet (+5%) gained noticeably.
3. **The chip stayed cool**, peaking at 68°C on the last model, much like Gen 2. Temperatures climb steadily through the run because the chip warms up as it goes, not because of the model.
4. **Gen 3 is worth keeping on.** It gives a free 8–15% cut in overall delay and was stable for the whole run.

**Current standing:** unchanged in ranking. YOLO11n still has the lowest overall delay among the useful models (14.9 ms), and DAMO-YOLO T (15.7 ms) moves close behind it with Gen 3. YOLO11s and DAMO-YOLO T remain the best balance of speed and accuracy.

**Added after this run: H10 YOLO26m** (`results/2026-10-24/l0_gen3_h10/`): 58.7 FPS (100% of Hailo's figure), 16.8 ms on the chip, 28.6 ms overall, 11.8 ms outside the chip. DAMO-YOLO M (47.8 mAP) was not added, because YOLO11m beats it on both accuracy and speed. YOLO11l and larger were not added either: chip time alone is 24 ms or more.

## L1 setup notes (2026-10-24)

- **Superseded (2026-10-24): decoding now covers all 80 classes.** The original note follows. **Decoding is person-only.** YOLO26 and DAMO-YOLO return raw tensors (YOLO11, YOLOv8 and NanoDet run NMS on the chip). Our decoders read only the person channel and apply the score threshold to the quantized values before converting anything. In the smoke test, post-processing took **0.2–0.3 ms for every family**, so the CPU decoding cost we expected for YOLO26 and DAMO-YOLO turned out to be negligible.
- **Decoder check:** on Ultralytics' `bus.jpg`, the YOLO26 and DAMO-YOLO boxes match YOLO11's to within a few pixels. The L1 run also saves annotated sample frames and a per-model agreement table against YOLO11m.
- **Preprocessing follows the Model Zoo configs:** YOLO models get a centered letterbox with grey (114) padding, DAMO-YOLO a top-left letterbox with black padding. Every model takes RGB input.
- **Camera:** it isn't fixed in place yet (it's handheld), so the orientation changes between recordings. `--rotate180` is opt-in; check the frames after every recording. Autofocus is continuous. It records 1536×864 at 30 fps; no recording so far has dropped a frame.
- **CPU load:** HailoRT runs three worker threads at about 2 ms of CPU each per frame. YOLO11n at 30 fps uses **28% of one core** in total, inside the 50% limit.
- **Python vs. L0:** inference through the Python API takes about 1 ms longer than `hailortcli`'s overall delay.
- **Pipelined mode** roughly doubles FPS without adding delay, but it also doubles CPU load.

## L1: recorded frames, person only (2026-10-24)

> **Superseded** by the all-class rerun below (*L1 rerun: all 80 classes*). Pre-processing and inference figures here still hold.

**Setup:** PCIe Gen 3, CPU governor `performance`, HailoRT 5.4.0 Python API. The input is 1000 recorded frames at 1536×864. They were recorded handheld and rotated upright afterwards, and the person appears only as a close-up in about half of them. Each model got 100 warm-up frames, then 1000 measured frames, repeated 3 times. Score threshold: 0.35. Throttling: none.

**Raw data:** `results/2026-10-24/l1_person_perf/` (`summary.csv`, `detections.csv`, per-frame timings in `frames/`, annotated frames in `samples/`). A partial run with the `ondemand` governor (H1–H3 only) is in `l1_person_ondemand_partial/`.

**One frame at a time** (in ms; median of the 3 repeats; "Worst" is the worst frame over all 3):

| ID | Model | mAP | Prep | Inference | Post-processing | **Total median** | **Total 99th pct** | Worst | CPU at 30 fps* |
|---|---|---|---|---|---|---|---|---|---|
| H9 | NanoDet (416) | 28.6 | 1.12 | 9.18 | 0.14 | 10.4 | 11.1 | 16.9 | ~6% |
| H3 | YOLO11n | 38.0 | 0.93 | 15.64 | 0.17 | **16.7** | **17.4** | 26.6 | ~24% |
| H5 | DAMO-YOLO T | 42.3 | 0.94 | 16.82 | 0.47 | 18.2 | 18.7 | 23.4 | ~27% |
| H1 | YOLO26n | 38.6 | 0.94 | 18.11 | 0.36 | 19.4 | 20.1 | 30.5 | ~27% |
| H4 | YOLO11s | 45.5 | 0.93 | 19.38 | 0.16 | 20.5 | 21.0 | 29.2 | ~24% |
| H6 | DAMO-YOLO S | 45.3 | 0.94 | 19.63 | 0.45 | 21.0 | 21.4 | 30.6 | ~27% |
| H2 | YOLO26s | 45.8 | 0.95 | 21.11 | 0.36 | 22.4 | 23.4 | 36.0 | ~27% |
| H8 | YOLO11m | 50.2 | 0.94 | 26.52 | 0.17 | 27.6 | 28.1 | 35.0 | ~24% |
| H10 | YOLO26m | 51.0 | 0.94 | 29.51 | 0.37 | 30.8 | 31.5 | 32.8 | ~27% |

\*Estimated from the CPU load at full speed, scaled to 30 fps. Direct measurement for YOLO11n gave 28%.

**Pipelined** (frame N+1 is prepared while frame N is on the chip):

| ID | Model | FPS one at a time → pipelined | Total median one at a time → pipelined (ms) | Total 99th pct pipelined (ms) |
|---|---|---|---|---|
| H3 | YOLO11n | 60 → 119 | 16.7 → 16.8 | 17.1 |
| H5 | DAMO-YOLO T | 55 → 106 | 18.2 → 18.8 | 19.7 |
| H1 | YOLO26n | 52 → 100 | 19.4 → 19.9 | 20.6 |
| H4 | YOLO11s | 49 → 97 | 20.5 → 20.5 | 21.0 |
| H8 | YOLO11m | 36 → 70 | 27.6 → 28.4 | 28.7 |
| H10 | YOLO26m | 32 → 59 | 30.8 → **34.0** | **38.0** |

**Governor check** (`ondemand` → `performance`, total delay one frame at a time): YOLO26n median 19.9 → 19.4 and 99th percentile 24.0 → 20.1; YOLO26s 99th percentile 25.0 → 23.4; YOLO11n 99th percentile 21.0 → 17.4 ms. With `performance`, the worst 1% of frames are 2–4 ms faster.

### What this shows

1. **Total delay is L0's overall delay plus about 2 ms.** Python inference adds about 0.8–1.2 ms over `hailortcli`, preparing the frame takes 0.94 ms, and post-processing takes 0.15–0.5 ms. After Gen 3, our own pipeline adds little delay; the rest of the ~11 ms outside the chip is spent in HailoRT and on PCIe.
2. **Delay is very steady.** The 99th percentile is within 0.5–1 ms of the median for every model. The worst frames (26–36 ms) are rare, single frames that only happen in one-frame-at-a-time mode. L4's 15-minute soak test should check whether they appear regularly.
3. **YOLO26 is about 2–3 ms slower than YOLO11 at the same size.** Decoding isn't the cause (0.36 ms). Its six raw 16-bit output tensors take longer to transfer and handle than YOLO11's finished on-chip boxes. Removing that would mean recompiling YOLO26 with its post-processing on the chip, which needs the x86 compiler.
4. **A 30 fps camera doesn't need pipelining.** Every model reaches at least 32 fps one frame at a time. Pipelining doubles FPS at almost no cost in delay, except for YOLO26m, where the chip is saturated: +3 ms and 38 ms at the 99th percentile. It's only needed for a 60 or 120 fps camera (comparison #5).
5. **CPU load is about 25% of one core at 30 fps for every model**, half the 50% limit. Most of it is HailoRT's worker threads, not our code.
6. **Detection sanity check:** every model except DAMO-YOLO T agrees with YOLO11m on 98.5–99.5% of frames, with box overlap (IoU) of 0.95–0.96 (NanoDet 0.89). **DAMO-YOLO T reports a person in 60% of frames instead of about 51%.** Most of its 95 extra frames are false alarms on clothes over a chair and on a purple cloth, with scores of 0.35–0.78, which a higher threshold won't fully remove. That risks locking onto the wrong object; L3 will measure it properly.
7. **Against the 40 ms budget:** L1 doesn't include camera time (exposure, readout, ISP), which L2 measures. Models with a 99th percentile around 21 ms or below (YOLO11n/s, DAMO-YOLO T/S, YOLO26n) leave about 20 ms for the camera. YOLO11m (28 ms) and YOLO26m (31.5 ms) leave only 9–12 ms and may fail once the camera is included.

**Current standing:** YOLO11n is the fastest useful model. **YOLO11s** gives s-size accuracy for +3.6 ms and is the most consistent model (std 0.18 ms). YOLO26 offers no advantage on this setup so far: it's slower at every size for about the same mAP. YOLO26m and YOLO11m stay in only if L3 shows the s models are too inaccurate on far-away people, and L2 must confirm they fit the budget.

## L1 rerun: all 80 classes (2026-10-24)

**Setup:** the same frames, settings and run rules as the person-only L1 run. Every model decodes all 80 COCO classes, with per-class NMS on the CPU for DAMO-YOLO. Throttling: none.

**Raw data:** `results/2026-10-24/l1_allclass/`

**One frame at a time** (ms; "Post" is the person-only figure → the all-class figure):

| ID | Model | Post median | Post 99th pct | **Total median** | **Total 99th pct** | Worst |
|---|---|---|---|---|---|---|
| H9 | NanoDet | 0.14 → 0.19 | 0.20 → 0.28 | 10.4 | 11.0 | 14.5 |
| H3 | YOLO11n | 0.17 → 0.21 | 0.20 → 0.25 | **16.8** | **17.3** | 24.6 |
| H5 | DAMO-YOLO T | 0.47 → 0.89 | 0.54 → 1.07 | 18.7 | 19.1 | 24.5 |
| H1 | YOLO26n | 0.36 → 0.78 | 0.43 → 0.88 | 19.9 | 20.3 | 30.0 |
| H4 | YOLO11s | 0.16 → 0.21 | 0.19 → 0.26 | **20.6** | **21.1** | 24.5 |
| H6 | DAMO-YOLO S | 0.45 → 0.89 | 0.50 → 1.08 | 21.5 | 22.1 | 34.5 |
| H2 | YOLO26s | 0.36 → 0.79 | 0.46 → 0.86 | 22.9 | 23.8 | 27.6 |

Pipelined mode gives the same post-processing times; the totals are 0.1–0.7 ms higher, as before.

**Detections compared with YOLO11m** (class-aware, IoU ≥ 0.5; YOLO11m is a reference, not ground truth):

| Model | Frames with a detection | Objects per frame | Share of YOLO11m's boxes it also finds | Share of its boxes YOLO11m confirms | Most frequent classes |
|---|---|---|---|---|---|
| YOLO26n | 74% | 0.86 | 45% | **85%** | person, tv, cup, mouse |
| YOLO26s | 76% | 0.97 | 50% | **84%** | person, laptop, cup, tv |
| YOLO11n | 78% | 1.00 | 46% | 75% | person, tv, **toothbrush (109)**, cup |
| YOLO11s | 82% | 1.38 | 62% | 73% | person, cup, laptop, bottle |
| DAMO-YOLO T | 93% | 1.62 | 62% | 64% | person (618), tv (384), laptop, cup |
| DAMO-YOLO S | 89% | 1.91 | **75%** | 64% | person, tv, cup, laptop |
| NanoDet | 89% | 1.37 | 51% | 61% | person, tv, cup, **train (93)** |

### What this shows

1. **All-class decoding costs little.** YOLO26 and DAMO-YOLO post-processing roughly doubles, to 0.8–0.9 ms at the median and about 1.1 ms at the 99th percentile, which adds 0.4–0.6 ms to the total. YOLO11, which uses on-chip NMS, adds only 0.05 ms. The ranking doesn't change. *(A 100-frame smoke test suggested up to 2.6 ms at the 99th percentile; the full run shows that was cold-start noise.)*
2. **For people, the models agree; for other objects, they don't.** All models find the person in about the same frames. For the room objects, overlap with YOLO11m is only 45–75%.
   - **YOLO26 is conservative:** most of its detections are confirmed (84–85%), but it finds only about half of YOLO11m's objects.
   - **DAMO-YOLO and NanoDet detect much more,** and many of those detections are unconfirmed (61–64%).
   - **Some repeated class labels look wrong:** YOLO11n calls something a "toothbrush" 109 times, and NanoDet sees a "train" 93 times.

   These are only signs. Real accuracy needs labelled clips (L3).
3. **The class-aware NMS in DAMO-YOLO** didn't add noticeable time over person-only decoding.

**Current standing:** unchanged. **YOLO11n and YOLO11s** are fastest, and their on-chip NMS makes all-class decoding nearly free. The extra decoding adds about 0.5 ms to the s models in L2, which were already 0.6–1.6 ms over the 40 ms budget at 60 fps. That makes the exposure and ISP-stream tests more important.

## L2: live camera, full pipeline (2026-10-24)

> **Person-only decoding:** this run decoded only the person class. Camera timing, exposure and the 30/60/120 fps conclusions still hold. From the all-class L1 rerun, YOLO26 and DAMO-YOLO totals should be about 0.4–0.6 ms higher and YOLO11 about 0.05 ms higher. The next L2 run (exposure test and ISP stream) decodes all classes.

**Setup:** imx708 NoIR camera at 1536×864, 30, 60 and 120 fps, with auto-exposure and continuous autofocus. PCIe Gen 3 and the `performance` governor. Each configuration ran 20 s after a 2 s warm-up, with stale queued frames flushed at the start. The camera pointed at an empty room, which doesn't matter for timing. The L4 sampler ran at 2 Hz throughout. Throttling: none (`0x0` before and after every run), no under-voltage, CPU held at 2400 MHz. Maximum temperatures: SoC 60.6 °C, Hailo 62.4 °C.

**Raw data:** `results/2026-10-24/l2_live/` (`summary.csv`, per-frame timestamps in `frames/`, 2 Hz system samples in `system/`).

**Clocks:** `SensorTimestamp` uses the same clock as `time.monotonic_ns()`. It marks the start of readout, not the exposure. A frame reaches Python 10.2–10.7 ms after it (median); about 10% of frames take 12–18 ms.

**Delay in pipelined mode** (ms). "From sensor timestamp" runs from the sensor timestamp to boxes ready. "From mid-exposure" adds half the exposure time, which is an estimate until the LED test:

| ID | Model | From sensor timestamp, median / 99th pct: 30 fps | 60 fps | 120 fps | From mid-exposure, 99th pct: 30 fps | 60 fps | 120 fps | CPU (one core) at 30 / 60 fps |
|---|---|---|---|---|---|---|---|---|
| H9 | NanoDet | 21.7 / 22.8 | 21.2 / 22.4 | 21.4 / 22.8 | 39.2 | 30.5 | 26.7 | 12% / 23% |
| H3 | YOLO11n | 27.8 / 29.0 | 28.8 / 30.0 | 30.9 / 36.3 | 45.4 | **38.1** | 40.3 | 30% / 58% |
| H5 | DAMO-YOLO T | 29.3 / 30.5 | 30.3 / 31.9 | 34.7 / 39.5 | 47.0 | **40.0** | 43.4 | 33% / 66% |
| H1 | YOLO26n | 30.3 / 32.1 | 31.1 / 32.5 | 35.9 / 40.2 | 48.5 | 40.6 | 44.2 | 33% / 65% |
| H4 | YOLO11s | 31.4 / 32.5 | 31.4 / 32.9 | 35.5 / 40.0 | 49.0 | 41.0 | 43.9 | 30% / 59% |
| H6 | DAMO-YOLO S | 32.1 / 33.2 | 32.5 / 33.5 | 38.0 / 42.3 | 49.6 | 41.6 | 46.2 | 33% / 65% |
| H2 | YOLO26s | 33.6 / 35.0 | 33.6 / 35.0 | 39.6 / 43.5 | 51.4 | 43.1 | 47.5 | 33% / 64% |
| H8 | YOLO11m | 38.6 / 40.0 | 38.1 / 39.4 | 43.1 / 47.1 | 56.5 | 47.5 | 51.0 | 30% / 59% |
| H10 | YOLO26m | 42.7 / 43.8 | 53.4 / 63.8 | 49.0 / 53.4 | 60.2 | 71.9 | 57.3 | 34% / 65% |

Auto-exposure used the full frame time indoors: 32.9 ms at 30 fps, 16.2 ms at 60 fps and 7.9 ms at 120 fps.

### What this shows

1. **Camera time is about 10.5 ms, plus half the exposure.** The camera's share of the delay is fixed for every model, and at 30 fps the exposure's share (about 16 ms) is larger than the chip's.
2. **60 fps + pipelined is the best mode for every model except YOLO26m.** Measured from the sensor timestamp, it's as fast as 30 fps, but halving the exposure saves about 8 ms of real delay, and results arrive at 60 Hz. Every model up to s size, and YOLO11m, keeps up with 60 fps without skipping frames.
3. **120 fps doesn't help, except for NanoDet.** No other model can process 120 frames per second, so the frame taken is up to one period old. The camera share rises to about 14–15 ms, which cancels the shorter exposure. 6–51% of frames are skipped.
4. **Pipelined mode is better even at 30 fps.** With one frame at a time, the worst frames reach 40–64 ms. Pipelined mode has the same median but a much tighter tail. YOLO26s, for example: 99th percentile 42.8 → 35.0 ms, worst frame 64.4 → 35.2 ms. At 60 fps, one-frame-at-a-time mode can't keep up at all (5–48% of frames skipped, 99th percentile 45–60 ms).
5. **Against the 40 ms budget, measured from mid-exposure at 60 fps:** YOLO11n (38.1) and DAMO-YOLO T (40.0) pass. YOLO26n, YOLO11s and DAMO-YOLO S are just over (40.6–41.6). YOLO26s (43.1), YOLO11m (47.5) and YOLO26m (64+) fail. Measured from the sensor timestamp, every model except YOLO26m passes.
6. **The m models are out.** YOLO26m can't keep up with 60 fps (the chip maxes out at 59) and goes over budget in every mode. YOLO11m fails by 7.5 ms at 60 fps, and even a very short exposure wouldn't close that gap.
7. **CPU load is over the limit at 60 fps.** It's 30–33% of one core at 30 fps, but 58–66% at 60 fps, over the 50% limit. Per frame that's about 10 ms of CPU: HailoRT about 6–7 ms, preparing the frame 1.4 ms (slower than in L1 because it reads from the camera buffer), and libcamera's work in the process.
8. **Temperature is no concern.** No throttling and no clock drops happened; the Hailo peaked at 62 °C.

**Current standing:** **YOLO11n and YOLO11s at 60 fps, pipelined** lead. DAMO-YOLO T is just as fast, but its false alarms (L1) have to be checked in L3. YOLO26 is still slower than YOLO11 at every size. The m models are dropped.

## L2: ISP-scaled stream and exposure (2026-10-24)

**Setup:** 60 fps, pipelined mode, all 80 classes. PCIe Gen 3, `performance` governor, no throttling (SoC ≤ 59.5 °C, Hailo ≤ 55.1 °C).
- **Stream run** (`results/2026-10-24/l2_stream/`): every model with auto-exposure, comparing the full 1536×864 stream with the ISP's 640×360 stream. With the ISP stream, preparing a frame is only a copy into the letterbox buffer.
- **Exposure run** (`results/2026-10-24/l2_exposure/`): YOLO11n and YOLO11s on the 640×360 stream, with exposure on auto, 8, 4 and 2 ms. Gain stays automatic.

**ISP stream** (auto-exposure, 16.2 ms at 16× gain in this room):

| ID | Model | Frame preparation (ms) | CPU (one core) | Total median from sensor timestamp (ms) | **From mid-exposure, 99th pct (ms)** |
|---|---|---|---|---|---|
| H9 | NanoDet | 1.49 → 1.07 | 24% → 20% | 21.5 → 20.7 | 31.0 → 30.2 |
| H3 | YOLO11n | 1.43 → 0.49 | 57% → **19%** | 28.5 → 26.6 | 37.6 → **36.1** |
| H5 | DAMO-YOLO T | 1.68 → 0.55 | 69% → 31% | 30.9 → 28.9 | 40.3 → **38.2** |
| H4 | YOLO11s | 1.35 → 0.49 | 58% → **18%** | 31.5 → 30.0 | 41.0 → **39.6** |
| H6 | DAMO-YOLO S | 1.37 → 0.49 | 68% → 27% | 33.2 → 31.6 | 42.4 → 41.0 |
| H2 | YOLO26s | 1.36 → 0.49 | 67% → 27% | 33.9 → 32.6 | 43.2 → 42.1 |
| H1 | YOLO26n | 1.40 → 0.56 | 74% → 30% | 32.5 → 30.0 | 50.9 → 43.7 |

(full 1536×864 stream → 640×360 ISP stream)

**Exposure sweep** (640×360 stream; gain was already at its 16× maximum, so a shorter exposure means a darker image):

| Exposure | Brightness (0–255) | Temporal noise | YOLO11n from mid-exposure, median / 99th pct | YOLO11s from mid-exposure, median / 99th pct |
|---|---|---|---|---|
| auto (16.2 ms) | 37 / 33 | 0.6–0.7 | 35.2 / 42.9 | 38.1 / 39.5 |
| 8 ms | 18–19 | 0.7 | 30.8 / 37.8 | 34.4 / 39.6 |
| 4 ms | 10 | 0.7 | 28.5 / 31.6 | 32.2 / 34.8 |
| 2 ms | 5 | 0.8 | 27.5 / 28.8 | 31.0 / 37.5 |

### What this shows

1. **The ISP stream cuts CPU load by two thirds.** YOLO11 drops to 18–19% of one core at 60 fps, and YOLO26 and DAMO-YOLO to 27–31%, all under the 50% limit. Frame preparation falls from about 1.4 to 0.5 ms, and the total drops by 1.3–2.5 ms. This should be the default from now on.
2. **With the ISP stream, YOLO11n, DAMO-YOLO T and YOLO11s pass the 40 ms budget** (measured from mid-exposure, 99th percentile). YOLO11s passes by only 0.4 ms. DAMO-YOLO S (41.0), YOLO26s (42.1) and YOLO26n (43.7) don't.
3. **Exposure doesn't change the time from sensor timestamp to boxes.** The camera share stays at about 10.1 ms at every exposure, which confirms the sensor timestamp marks the start of readout. Each millisecond less exposure saves exactly 0.5 ms of real delay: an 8 ms exposure saves about 4 ms over auto, and 4 ms saves about 6 ms.
4. **In this room, short exposures aren't usable.** Gain was already at its maximum, so brightness falls in proportion to exposure: 37 → 18 → 10 → 5 out of 255. Below about 8 ms the image is close to black. Shorter exposure only helps in brighter light. Auto-exposure shortens it on its own in bright light, or IR lighting can help, since this is a NoIR camera.
5. **The 99th percentile varies by 3–7 ms between 20-second runs.** YOLO11n with the same settings measured 28.0 ms in the stream run and 34.8 ms in the exposure run, from the sensor timestamp. YOLO11s's 0.4 ms margin is therefore within run-to-run noise. Deciding between YOLO11n and YOLO11s needs the 15-minute soak test (L4) and the LED test.
6. **The exposure run's detection counts aren't usable.** The camera faced an out-of-focus desk edge with no clear objects in view (YOLO11n: 98.5% of frames on auto, YOLO11s: 0%). Image quality against exposure has to be measured on a proper scene in L3.

**Current standing:** **YOLO11s at 60 fps with the 640×360 ISP stream in pipelined mode**, with **YOLO11n** as the fallback. YOLO11s is the most accurate model that passes, it has the lowest CPU load (18%), and it does NMS on the chip, which makes decoding all classes almost free. Its margin is very thin, though. The final choice depends on the LED test, the 15-minute soak test and accuracy on your own clips (L3).

## Decision (2026-10-24): YOLO26n for now

**Chosen detector for building the rest of the system: H1 YOLO26n on the Hailo-10H**, at 60 fps with the 640×360 ISP stream in pipelined mode, decoding all 80 classes. This choice is provisional, and the detector comparison continues later (LED test, soak test, L3 accuracy).

YOLO26n's figures in that setup, which go into the simulator's latency model: 30.0 ms median and 35.6 ms 99th percentile from the sensor timestamp to boxes; about 43.7 ms at the 99th percentile counted from mid-exposure, with indoor auto-exposure of 16 ms; 30% of one CPU core. That's over the draft 40 ms budget. YOLO11s (39.6 ms) and YOLO11n (36.1 ms) remain the candidates if the budget turns out to matter. All three use the same code path (`hailo_detect.py`), so switching only means changing the HEF.

### Next
- **LED test** (`led_latency.py`): the real delay from a light turning on to boxes ready, to check the "from mid-exposure" estimate.
- **L4 soak test:** YOLO11s and YOLO11n for 15 minutes each with the final settings, for a reliable 99th percentile and worst case, plus any drift as the board heats up.
- **Lighting:** test with more light or an IR illuminator, so auto-exposure picks a shorter exposure.
- Keep PCIe Gen 3 and the `performance` governor on for all further tests.
- L3: record proper clips with the camera fixed in place, with several object types (people and everyday objects) at near, middle and far, standing still, slow and fast movement, occlusion, and empty scenes. Label all classes.
