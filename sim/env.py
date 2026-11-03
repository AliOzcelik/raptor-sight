"""Gymnasium wrapper: one BatchSim env as a gymnasium.Env (for Stable-Baselines3 and checks).

For many envs at once, use sim.core.BatchSim directly and reset finished envs with reset(done).
"""

import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from sim.config import SimConfig
from sim.core import BatchSim
from sim.obs import OBS_NAMES


class PanTiltEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 60}

    def __init__(self, cfg = None, render_mode = None):
        self.cfg = cfg or SimConfig()
        self.render_mode = render_mode
        self.sim = BatchSim(1, self.cfg)
        self.observation_space = spaces.Box(-1.0, 1.0, (len(OBS_NAMES),), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (2,), np.float32)
        self.info = {}


    def reset(self, *, seed = None, options = None):
        super().reset(seed = seed)
        if seed is not None:
            self.sim.seed(seed)
        if options and "difficulty" in options:
            self.sim.difficulty = float(options["difficulty"])
        obs = self.sim.reset()[0].copy()
        self.info = {}
        return obs, {}


    def step(self, action):
        obs, reward, terminated, truncated, info = self.sim.step(np.asarray(action, np.float64)[None])
        self.info = {k: v[0] for k, v in info.items()}
        if self.render_mode == "human":
            self.render()
        return obs[0].copy(), float(reward[0]), bool(terminated[0]), bool(truncated[0]), self.info


    def render(self):
        """Camera view: true target box (green), detector box (yellow, when found), tracker estimate (red)"""
        s, W, H = self.sim, self.sim.W, self.sim.H
        img = np.full((H, W, 3), 40, np.uint8)
        cv2.drawMarker(img, (W // 2, H // 2), (255, 255, 255), cv2.MARKER_CROSS, 20, 1)
        mount = np.column_stack([s.p["mount_pan_deg"], s.p["mount_tilt_deg"]])
        box, front = s.project(s.out + mount, s.tgt, np.arange(1))
        if front[0]:
            cv2.rectangle(img, tuple(box[0, :2].astype(int)), tuple(box[0, 2:].astype(int)), (0, 200, 0), 2)
        if s.last["found"][0]:
            b = s.last["box"][0].astype(int)
            cv2.rectangle(img, tuple(b[:2]), tuple(b[2:]), (0, 220, 220), 1)
        e = s.last["e"][0]
        p = (int(W / 2 * (1 + e[0])), int(H / 2 * (1 - e[1])))
        cv2.circle(img, p, 6, (0, 0, 255), -1 if s.last["valid"][0] else 2)
        cv2.putText(img, f"t {s.t[0]:5.2f} s  pan {s.out[0, 0]:+6.1f}  tilt {s.out[0, 1]:+6.1f}"
                         f"  latency {s.last['latency'][0] * 1e3:.0f} ms",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        if self.render_mode == "human":
            cv2.imshow("sim", img)
            cv2.waitKey(1)
        return img


    def close(self):
        if self.render_mode == "human":
            cv2.destroyWindow("sim")
