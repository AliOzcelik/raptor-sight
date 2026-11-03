# RaptorSight — Sim-to-Real RL Pan-Tilt Object Tracker

> **Sim-to-real reinforcement learning for real-time pan-tilt object tracking on Raspberry Pi 5 + Hailo-10H.**

| | |
|---|---|
| **Status** | Design complete, hardware partially in hand, implementation starting |
| **Name** | RaptorSight (repo slug `raptorsight`) |
| **Author** | Ali Özçelik ([@AliOzcelik](https://github.com/AliOzcelik)) |
| **Target platform** | Raspberry Pi 5 + AI HAT+ 2 (Hailo-10H NPU) + Pi camera |
| **Actuation** | 2× hobby servos (pan + tilt), optional ESP32-S3 real-time driver |
| **RL algorithm** | SAC (current choice, may change — TD3 / PPO are alternatives) |

---

## 1. Overview

RaptorSight is a camera on a two-axis pan-tilt mount that keeps a detected object centered in the frame. Instead of a hand-tuned PID controller, the control policy is a small neural network trained with **deep reinforcement learning entirely in simulation**, then transferred to real hardware with **no or minimal real-world fine-tuning**.

The project's core loop:

```
perceive (camera + Hailo detector) → observe (bbox error) → policy (MLP) → act (pan/tilt command) → servos move → repeat
```

### Goals

1. Train a tracking policy fully in a lightweight simulator.
2. Transfer it to real hardware using domain randomization, system identification, and explicit latency modeling.
3. Run the whole stack on the edge: detection on the Hailo-10H NPU, policy on the Pi 5 CPU (or an ESP32).
4. **Prove value** by beating a well-tuned PID baseline on tracking error, target-loss rate, and smoothness.

### Non-goals

- Pixel-level (image-based) RL. The policy works on an abstract, detector-level state, which makes sim-to-real far easier.
- High-fidelity physics simulation as a requirement (MuJoCo/PyBullet are optional upgrades only).

---

## 2. System Architecture

```
┌─────────────────────────── Raspberry Pi 5 ───────────────────────────┐
│                                                                      │
│  Pi Camera ──► Capture ──► Hailo-10H detector ──► Target selection   │
│                (Picamera2)   (YOLO-class, HEF)     + smoothing/flags  │
│                                                         │            │
│                                                         ▼            │
│                                      Observation builder (normalize) │
│                                                         │            │
│                                                         ▼            │
│                                         RL policy (ONNX, CPU, MLP)   │
│                                                         │            │
└─────────────────────────────────────────────────────────┼────────────┘
                                                          │
                     ┌────────────────────────────────────┴───────────┐
                     ▼                                                ▼
          Variant A: Pi path                              Variant B: ESP32 path
          I²C → PCA9685 → servos                          UART → ESP32-S3 → PWM → servos
                                                          (deterministic timing,
                                                           optional on-MCU policy)
```

### Component responsibilities

1. **Perception:** Hailo-10H runs the object detector and returns bounding boxes. The target can be any object class (not fixed yet).
2. **State estimation:** Selects the target box, normalizes it, computes error and error velocity, and flags missing detections.
3. **Policy:** A small MLP maps the observation to pan/tilt commands. It runs at camera frame rate (~30 Hz).
4. **Actuation:** Converts normalized actions into servo pulse widths, with angle limits and rate limits as a safety layer.

---

## 3. MDP Design

### 3.1 Observation (policy input)

All values normalized to roughly `[-1, 1]`.

| Feature | Description |
|---|---|
| `e_x, e_y` | Target center error from image center, normalized by half-width/height |
| `ė_x, ė_y` | Error velocity (finite difference, filtered) |
| `w, h` | Normalized bbox size (proxy for distance and speed in image space) |
| `a_{t-1}` | Previous action (both axes), helps the policy learn smoothness and compensate latency |
| `θ_pan, θ_tilt` | Commanded servo angles (MG90S has no encoder feedback) |
| `valid` | 1 if a detection exists this frame, else 0 |
| `t_lost` | Normalized time since last valid detection |

A short **frame stack** (e.g. last 3 observations) is optional and helps under latency.

### 3.2 Action

- Continuous, 2-D: `a ∈ [-1, 1]²`.
- Interpreted as **angular velocity / delta-angle commands** per step, scaled by a max rate (e.g. ±X°/step).
- Final command is clipped to mechanical limits and passed through a rate limiter.

### 3.3 Reward

```
r = − w_e · ‖e‖²            (centering)
    − w_s · ‖a_t − a_{t−1}‖²  (smoothness → prevents servo chatter/jitter)
    − w_a · ‖a_t‖²           (effort)
    + w_c · 1[‖e‖ < ε]       (bonus for being centered)
    − w_l · 1[target lost]   (large penalty for losing the target)
```

The smoothness term is essential: without it, policies learn bang-bang behavior that looks fine in sim and destroys servos in real life.

### 3.4 Termination / truncation

- **Terminate:** target outside the frame longer than a threshold.
- **Truncate:** fixed episode length (time limit).

---

## 4. Simulator

A lightweight, analytic, **bbox-level** simulator, Gymnasium-compatible and vectorizable (thousands of parallel envs).

### 4.1 Components

1. **Pan-tilt kinematics:** two revolute joints with angle limits.
2. **Servo dynamics:** first-order lag, max slew rate, deadband, backlash, quantization.
3. **Camera model:** pinhole projection with configurable FOV and resolution; target world position → pixel coordinates and bbox size.
4. **Target trajectories:** random walk, sinusoidal, sudden direction changes, acceleration bursts, stop-and-go, temporary occlusion.
5. **Latency pipeline:** queues that delay both observations and actions (camera exposure + inference + communication + actuation).
6. **Detector noise model:** Gaussian jitter on bbox center/size, false negatives (dropout), occasional false positives, FPS jitter.

### 4.2 Optional upgrades

- MuJoCo or PyBullet for physical realism (mount inertia, vibration) if the analytic model proves insufficient.

---

## 5. Sim-to-Real Bridge (the core of the project)

### 5.1 Domain randomization

Randomized per episode:

| Parameter | Why |
|---|---|
| Servo speed, lag time constant | Real servos vary unit to unit and with load/voltage |
| Deadband, backlash | Cheap gears (MG90S) have significant play |
| End-to-end latency + jitter | The #1 cause of sim-to-real failure in visual servoing |
| Detection noise, dropout rate | Real detectors flicker, especially on small/fast targets |
| FPS jitter | Inference time is not constant |
| Camera FOV, mounting offset | Calibration error |
| Target speed / trajectory family | Generalization to unseen motion |

### 5.2 System identification

1. Measure real servo **step responses** (command a step, record the angle over time).
2. Standard servos lack position feedback, so angles are measured with an **ArUco marker** on the mount viewed by a second camera (or the main camera against a fixed marker).
3. Fit lag, slew rate, deadband, and latency, then **center the randomization ranges** on the measured values.

### 5.3 Latency modeling

Actions are applied with an explicit delay in sim. The real end-to-end latency (capture → detection → policy → PWM → motion) is measured on hardware and used to set the delay distribution.

### 5.4 Asymmetric actor-critic

- **Actor** sees only what is available on the real robot (noisy, delayed observations).
- **Critic** additionally sees privileged sim state: true target position/velocity, true servo angles, current latency and randomization parameters.
- This improves value estimates and training stability under heavy randomization, and the critic is discarded at deployment.

---

## 6. Training

| Item | Choice |
|---|---|
| Algorithm | **SAC** (sample-efficient, continuous actions, robust exploration). TD3 and PPO are alternatives; PPO scales well with massive parallel envs. |
| Network | Small MLP (e.g. 2×256 actor), small enough for Pi CPU and ESP32 |
| Environment | Vectorized analytic sim, fully randomized |
| Curriculum | Start with slow targets and low noise, increase speed, latency, and dropout over training |
| Real-world fine-tuning | Optional, only if the sim-to-real gap remains after system identification |

### Training outputs

- Policy checkpoint (PyTorch)
- Exported actor: **ONNX** (Pi) and **INT8 TFLite Micro** (ESP32 variant)
- Normalization statistics (must be shipped with the policy)

---

## 7. Deployment

### 7.1 Perception on Hailo-10H

- Detector: a small YOLO-class model, compiled to **HEF for Hailo-10H**.
- ⚠️ The AI HAT+ 2 uses the **Hailo-10H**, not the Hailo-8. Pre-compiled Hailo-8 HEF files do not run on it, so models must be compiled or downloaded for the Hailo-10H.

### 7.2 Policy runtime

- **Pi path:** ONNX Runtime on the Pi 5 CPU. A small MLP runs in well under a millisecond.
- **ESP32 path (optional):** the policy runs as an INT8 TFLite Micro model on the ESP32-S3, with the Pi only sending observations.

### 7.3 Actuation variants

| Variant | How it works | Pros | Cons |
|---|---|---|---|
| **A. Pi path** | Pi → I²C → PCA9685 → servos | Simple, fewest parts | Linux timing jitter |
| **B. ESP32 path** | Pi → UART → ESP32-S3 → PWM | Deterministic real-time PWM, rate limiting at MCU level | Extra firmware, serial protocol |

### 7.4 Safety layer (always on)

- Hard angle limits per axis
- Command rate limiting
- Watchdog: servos hold position if commands stop arriving
- Behavior on target loss: hold, then slow search scan

---

## 8. Hardware

### 8.1 Already owned

- Raspberry Pi 5
- AI HAT+ 2 (Hailo-10H)
- Pi camera (connected)

### 8.2 Bill of materials (remaining)

| Part | Qty | Purpose |
|---|---|---|
| Pan-tilt bracket (SG90/MG90S size) | 1 | Mechanical mount |
| MG90S metal-gear servos | 2 | Pan and tilt actuation |
| PCA9685 16-channel PWM driver | 1 | Servo control over I²C (Variant A) |
| 5–6 V, ≥3 A power supply | 1 | Dedicated servo power (never from Pi 5V rail) |
| 1000 µF electrolytic capacitor | 1 | Absorbs servo current spikes |
| Longer camera ribbon cable | 1 | Lets the camera move with the mount |
| ESP32-S3-DevKitC-1 | 1 | Variant B real-time driver |
| Jumper wires, common ground | — | Wiring |

### 8.3 Upgrade option: bus servos with feedback

- **Waveshare ST3215** serial bus servos + driver board.
- Built-in **encoder feedback** (true angle readout) → much better system identification, real joint angles in the observation, less backlash.

### 8.4 Wiring rules

1. Servos get their own supply; share **ground** with the Pi/ESP32.
2. Capacitor across the servo supply, close to the driver.
3. Secure the camera cable so it doesn't fight the tilt axis.

---

## 9. Evaluation

### 9.1 Metrics

| Metric | Meaning |
|---|---|
| Mean / max pixel error | Tracking accuracy |
| Target-loss rate | Robustness (fraction of time target out of frame) |
| Re-acquisition time | Recovery after loss |
| Settling time | Response to step changes |
| Action smoothness (jerk, Σ‖Δa‖) | Servo wear, video quality |
| End-to-end latency | System performance |

### 9.2 Baseline

A **well-tuned PID controller** (per axis, with derivative filtering and anti-windup) evaluated on identical scenarios. RL's value is only proven by this comparison.

### 9.3 Test protocol

1. **Sim:** held-out trajectory families and randomization ranges wider than training.
2. **Real:** repeatable target motion (e.g. a target on a moving rig or screen) recorded and replayed for RL vs PID.
3. **Ablations:** no domain randomization, no latency modeling, symmetric critic.

---

## 10. Repository Structure

```
raptorsight/
├── sim/          # Gymnasium env, servo/camera/latency models, trajectories
├── rl/           # training scripts, configs, asymmetric actor-critic
├── perception/   # Hailo-10H detector integration, target selection
├── deploy/
│   ├── pi/       # runtime loop, ONNX policy, PCA9685 driver
│   └── esp32/    # firmware, serial protocol, TFLite Micro policy
├── hardware/     # wiring diagrams, BOM, mount notes
├── sysid/        # step-response capture, ArUco angle measurement, fitting
├── eval/         # metrics, PID baseline, RL-vs-PID reports
├── configs/      # randomization ranges, reward weights, hardware params
└── docs/
```

---

## 11. Roadmap

| Phase | Milestone | Done when |
|---|---|---|
| **0** | Hardware bring-up | Servos move safely via PCA9685; camera + Hailo detector stream boxes |
| **1** | Simulator v1 | Gymnasium env with kinematics, camera, trajectories, latency |
| **2** | PID baseline | PID tracks in sim and on hardware; metrics logged |
| **3** | RL in sim | SAC beats PID in sim on nominal parameters |
| **4** | System identification | Servo lag, deadband, latency measured; sim calibrated |
| **5** | Robust training | Domain randomization + asymmetric critic + curriculum |
| **6** | Sim-to-real transfer | ONNX policy tracks on the Pi without fine-tuning |
| **7** | Evaluation | RL vs PID report on real hardware, ablations |
| **8** | ESP32 variant | Real-time PWM path (and optionally on-MCU policy) |
| **9** | Extensions | See below |

---

## 12. Future Extensions

1. **Auto-cameraman:** reward cinematic framing (rule of thirds, headroom) instead of pure centering.
2. **Wildlife / bird-feeder camera:** small, erratic targets, species classification, automatic clip recording, scan → lock re-acquisition behavior.
3. **Language-commanded tracking:** open-vocabulary detector + text-embedding-conditioned policy ("follow the red cup"), exposed as an agent tool `track(target_description)`.
4. **Ball / sports tracking:** predictive leading of fast targets to compensate latency, with shot analytics.
5. **Bus-servo upgrade:** encoder feedback in the observation for tighter control.

---

## 13. Key Risks

| Risk | Mitigation |
|---|---|
| Latency mismatch sim vs real | Measure end-to-end latency; randomize around it; action history in obs |
| Servo backlash / deadband | System ID; randomization; bus-servo upgrade |
| Detector flicker | Dropout modeling; `valid` / `t_lost` features; light filtering |
| Jittery policy | Smoothness reward; rate limiter; action filtering |
| Hailo-10H model compatibility | Use Hailo-10H-specific HEF models only |
| Brownouts from servos | Separate supply, capacitor, common ground |

---

## 14. Tech Stack

- **Language:** Python (training, sim, Pi runtime), C++ (ESP32 firmware)
- **RL:** Gymnasium, PyTorch, Stable-Baselines3 (or custom SAC)
- **Perception:** Hailo runtime / Hailo apps, Picamera2, OpenCV (ArUco)
- **Deployment:** ONNX Runtime (Pi), TFLite Micro (ESP32)
- **Hardware I/O:** PCA9685 over I²C, UART serial to ESP32
