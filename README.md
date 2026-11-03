# SungurTrack (RaptorSight)

**Sim-to-real reinforcement learning for real-time pan-tilt object tracking on a Raspberry Pi 5 with the Hailo-10H AI accelerator.**

## Goal

The project builds a camera on a two-axis pan-tilt mount that keeps a detected object centred in the frame. The target can be any object the detector knows; there is no fixed class.

Instead of a hand-tuned controller, the mount is steered by a small neural network policy trained with reinforcement learning entirely in a simulator. The policy is then moved to the real hardware with little or no real-world training. To get there, the simulator models the parts where simulation and reality usually differ: latency, servo imperfections and detector noise. These values are randomized during training and calibrated against measurements from the real system.

The policy's value is judged against a well-tuned PID controller on the same scenarios, by:
- tracking error,
- target-loss rate,
- re-acquisition time,
- smoothness.

```
camera → detector → tracker → observation → controller (PID or RL policy) → pan/tilt servos
```

The full design is in [PROJECT.md](PROJECT.md).

## Hardware

| Device | Role | Status |
|---|---|---|
| Raspberry Pi 5 | Runs the real-time loop: capture, tracking, controller | In use |
| Raspberry Pi AI HAT+ 2 (Hailo-10H) | Object detection | In use (PCIe Gen 3) |
| Raspberry Pi Camera Module 3 NoIR (Sony IMX708) | Tracking camera, 640×360 at 60 fps | In use (not mounted yet) |
| 2× MG90S servos + pan-tilt bracket | Pan and tilt movement | Planned |
| PCA9685 PWM driver | Servo control over I²C from the Pi | Planned |
| 5–6 V, ≥3 A supply + 1000 µF capacitor | Separate servo power | Planned |
| ESP32-S3 | Optional real-time servo driver over UART | Planned (optional) |

Only the camera and the detector run on real hardware so far. Everything after the tracker has been tested in simulation only.

## What has been done

### 1. Detector benchmark on the Hailo-10H
Ten pre-compiled Hailo-10H models were benchmarked at three levels:

| Level | What it measures |
|---|---|
| L0 | Chip speed on its own |
| L1 | Speed on 1000 recorded camera frames |
| L2 | The full live pipeline from the camera |

The models: YOLO26 n/s/m, YOLO11 n/s/m, YOLOv8m, DAMO-YOLO T/S and NanoDet. The full plan and results are in [perception/bench/plan.md](perception/bench/plan.md).

Main findings:
- **PCIe Gen 3** cuts 1.6–3 ms from every model's delay compared with Gen 2.
- **Pipelined mode:** preparing the next frame while the current one is on the chip doubles throughput at almost no added delay.
- **60 fps with the camera ISP's 640×360 stream** is the best mode. The ISP stream cuts CPU load by about two thirds.
- **The camera itself adds about 10 ms plus half the exposure time.**
- **The m-size models are too slow** for a 60 fps loop.

**Chosen detector (provisional): YOLO26n** at 60 fps, pipelined, decoding all 80 COCO classes. In the benchmark it measured 30.0 ms median and 35.6 ms 99th percentile from the sensor timestamp to boxes. The CPU load was about 30% of one core.

### 2. Live detection pipeline
`raspberrypi_camera.py` runs YOLO26n on the live camera feed at 60 fps, with a depth-2 pipeline that always takes the newest frame. Each result carries:
- the sensor timestamp and a frame number that shows skipped frames;
- exposure, gain and latency;
- boxes in pixels, and pan/tilt angles of the box centres. The angles use the FOV from the camera datasheet, which hasn't been calibrated yet.

The tracker is connected to it, with a display, click-to-lock and CSV logging. A 57-second live run logged 60 fps with no skipped frames and a latency of 38.4 ms median / 42.8 ms 99th percentile, from mid-exposure to the tracker's output.

### 3. Target tracking
`perception/target.py` turns each frame's boxes into the controller's input:
- **Target selection:** picks one box (the one closest to the centre, or a clicked one) and keeps following the same object by overlap and class.
- **State estimation:** an alpha-beta filter gives the error velocity from the camera timestamps, so skipped frames are handled correctly.
- **Lost targets:** while the target is missing, it predicts the position and reports how long it has been lost.

`perception/analyze_track.py` summarizes the tracking logs: timing, dropouts, target switches, detector jitter, and velocity error and lag.

### 4. Simulator
`sim/` is a vectorized, box-level simulator with a Gymnasium interface. It models:

| Part | What's simulated |
|---|---|
| Pan-tilt | Two joints with angle limits |
| Servos | Command delay, 20 ms PWM frames, quantization, deadband, first-order lag, slew limit, backlash |
| Camera | Pinhole projection with a mounting offset |
| Targets | Six motion types (still, random walk, sinusoid, direction changes, bursts, stop-and-go), plus occlusions |
| Detector | Box jitter, size noise, missed and false detections, minimum size, clipping at the frame edge |
| Latency | Measured from the real system, with jitter and skipped frames |
| Tracker | An exact copy of the tracker running on the Pi |

The observation, reward and termination follow PROJECT.md. All 26 parameters are randomized per episode, with one difficulty setting for a curriculum. Privileged state for an asymmetric critic is also provided.

Latency, frame rate, detector noise and dropouts are based on measurements from the real system. The servo parameters are MG90S estimates.

The simulator runs at about 106,000 environment-steps per second with 1024 environments on the Pi 5.

### 5. PID baseline (in simulation)
`eval/` contains:
- **`pid.py`:** a per-axis PID controller with anti-windup and lost-target handling, using only the information available on the real system.
- **`evaluate.py`:** fixed test scenarios and the evaluation metrics.
- **`tune_pid.py`:** a gain search in the simulator.

Tuned gains: kp 0.188, ki 0.112, kd 0. On held-out test scenarios:

| Controller | Mean score | Step test settling time |
|---|---|---|
| No action | −0.700 | does not settle |
| P controller (gain 0.3) | −0.236 | 1.05 s |
| **Tuned PID** | **−0.084** | **0.40 s** |

With nominal parameters, the tuned PID keeps the target within about 4–11 px and never loses it. With full randomization, it fails 9–50% of the 20-second episodes, depending on the motion type. This is the baseline the RL policy has to beat.

## Repository layout

```
perception/
  bench/             detector benchmark scripts, model list, plan and results
  target.py          target selection and state estimation
  analyze_track.py   tracking log analysis
raspberrypi_camera.py  live detection and tracking on the Pi
sim/                 simulator (config, dynamics, trajectories, observation, Gymnasium env, checks)
eval/                PID baseline, test scenarios, metrics, tuning, results
test/                camera image check
PROJECT.md           full project design
```
