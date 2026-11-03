#!/usr/bin/env python3
"""Checks for the simulator: Gymnasium API, the tracker copy against perception/target.py,
servo and latency behaviour, a P controller against doing nothing, and speed.

    .venv/bin/python -m sim.check_sim
"""

import dataclasses
import time
from types import SimpleNamespace

import numpy as np
from gymnasium.utils.env_checker import check_env

from perception.target import Tracker, TrackerConfig
from sim.config import SimConfig
from sim.core import BatchSim
from sim.env import PanTiltEnv
from sim.obs import BatchEstimator, OBS_NAMES


def check_tracker(n_frames = 3000, seed = 0):
    """Same detections through perception/target.py and sim.obs.BatchEstimator -> same outputs"""
    rng = np.random.default_rng(seed)
    cfg = TrackerConfig()
    ref, est = Tracker(cfg), BatchEstimator(1, cfg)
    c, t, worst = np.array([320.0, 180.0]), 0, 0.0
    gap = 0
    for k in range(n_frames):
        t += int(1e9 / 60) * (2 if rng.random() < 0.02 else 1)
        c = (c + rng.normal(0, 4, 2)).clip(40, [600, 320])
        if gap == 0 and rng.random() < 0.02:
            gap = int(rng.choice([2, 10, 25, 40]))      # short and longer than lose_after_s
        jump = rng.random() < 0.01                       # a box far away: must not match
        found = gap == 0
        gap = max(0, gap - 1)
        cc = c + (rng.uniform(150, 250, 2) * rng.choice([-1, 1], 2) if jump else 0)
        box = np.array([[cc[0] - 30, cc[1] - 40, cc[0] + 30, cc[1] + 40]])
        dets = SimpleNamespace(frame = SimpleNamespace(t_exposure_ns = t, frame_id = k),
                               boxes = np.column_stack([box, [[0.9, 0]]]).astype(np.float32) if found
                               else np.zeros((0, 6), np.float32))
        o = ref.update(dets)
        e, de, size, valid, t_lost = est.update(box, np.array([found]), np.array([t / 1e9]), np.array([0]))
        a = np.array([o.e_x, o.e_y, o.de_x, o.de_y, o.w, o.h, o.valid, o.t_lost])
        b = np.concatenate([e[0], de[0], size[0], valid, t_lost])
        worst = max(worst, np.abs(a - b).max())
    print(f"tracker copy vs perception/target.py: max difference {worst:.2e} over {n_frames} frames")
    assert worst < 1e-3


def servo_step(cfg):
    """Command a 30 deg pan step with nominal parameters; time until the camera reaches 90 %"""
    sim = BatchSim(1, dataclasses.replace(cfg, difficulty = 0.0), seed = 1)
    sim.out[:] = sim.motor[:] = sim.cmd[:] = sim.cmd_eff[:] = sim.latched[:] = sim.pending[:] = 0.0
    a = np.array([[1.0, 0.0]])
    t, hist = 0, []
    for k in range(60):
        sim.step(a if k < 10 else np.zeros((1, 2)))      # 10 steps of +3 deg
        hist.append((sim.t[0], sim.out[0, 0]))
    hist = np.array(hist)
    t90 = hist[np.argmax(hist[:, 1] >= 27.0), 0]
    print(f"servo: 30 deg commanded over 10 steps ({10 / 60 * 1e3:.0f} ms); camera at 90 % after {t90 * 1e3:.0f} ms,"
          f" final {hist[-1, 1]:.2f} deg (backlash, deadband and quantization keep it off 30.00)")


def run_policy(cfg, policy, n = 256, seconds = 20.0, seed = 0):
    sim = BatchSim(n, cfg, seed = seed)
    obs = sim.obs
    steps = int(seconds * cfg.fps)
    err, in_frame, rew, done_term, lat, valid = [], [], [], 0, [], []
    for _ in range(steps):
        obs, r, term, trunc, info = sim.step(policy(obs, sim))
        err.append(np.linalg.norm(info["e_true"], axis = 1))
        in_frame.append(info["in_frame"])
        rew.append(r)
        lat.append(sim.last["latency"].copy())
        valid.append(obs[:, OBS_NAMES.index("valid")])
        done_term += term.sum()
        obs = sim.reset(term | trunc)
    err, in_frame = np.array(err), np.array(in_frame)
    return dict(err = float(err[in_frame].mean()), in_frame = 100 * float(in_frame.mean()),
                reward = float(np.mean(rew)), terminations = int(done_term), latency_ms = 1e3 * float(np.mean(lat)),
                valid = 100 * float(np.mean(valid)))


def p_controller(gain):
    """Turn by gain x the observed error angle per step (needs to stay small: ~3 frames of latency)"""
    def policy(obs, sim):
        hfov = sim.p["hfov_deg"]
        half_fov = np.column_stack([hfov, hfov * sim.H / sim.W]) / 2
        return np.clip(gain * obs[:, :2] * half_fov / sim.cfg.servo.max_step_deg, -1, 1)
    return policy


def main():
    cfg = SimConfig()
    check_env(PanTiltEnv(cfg), skip_render_check = True)
    print("gymnasium check_env: ok")
    check_tracker()
    servo_step(cfg)

    print("\n20 s on 256 envs (finished envs restart):")
    for d in (0.0, 0.5, 1.0):
        c = dataclasses.replace(cfg, difficulty = d)
        for name, pol in (("no action", lambda o, s: np.zeros((s.n, 2))), ("P 0.15", p_controller(0.15)),
                          ("P 0.3", p_controller(0.3)), ("P 0.6", p_controller(0.6))):
            r = run_policy(c, pol)
            print(f"  difficulty {d:.1f}  {name:9s}  |e| {r['err']:.3f} while in frame, in frame {r['in_frame']:5.1f} %,"
                  f" valid {r['valid']:5.1f} %, reward/step {r['reward']:+.3f}, terminations {r['terminations']:3d},"
                  f" latency {r['latency_ms']:.1f} ms")

    env = PanTiltEnv(cfg)
    env.reset(seed = 0)
    t0, k = time.perf_counter(), 0
    while time.perf_counter() - t0 < 2.0:
        _, _, term, trunc, _ = env.step(env.action_space.sample())
        k += 1
        if term or trunc:
            env.reset()
    print(f"\nspeed: gymnasium env {k / (time.perf_counter() - t0):,.0f} steps/s", end = "")
    sim = BatchSim(1024, cfg, seed = 0)
    a = np.zeros((1024, 2))
    t0 = time.perf_counter()
    for _ in range(100):
        sim.step(a)
    print(f", BatchSim x1024 {100 * 1024 / (time.perf_counter() - t0):,.0f} env-steps/s")


if __name__ == "__main__":
    main()
