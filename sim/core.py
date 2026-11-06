"""Vectorized pan-tilt tracking simulator: N independent envs stepped together with NumPy.

One step is one camera frame result, as in raspberrypi_camera.py:
  t_k          observation k is ready; the controller answers with action a_k at once
  + act        the new commanded angle reaches the servo driver (policy + I2C)
  next PWM     the servo latches it, quantized, at its next PWM frame
  ...          servos move (lag, slew limit, deadband, backlash), the target moves
  t_k+1        observation k+1 is ready. It shows the scene at its exposure time (middle of the
               exposure) through the detector model and the same tracker as on the Pi.
The reward uses the true target position at t_k+1.

Timing follows the Pi: frames are exposed on the sensor's regular clock (one frame period apart,
two when a frame is skipped) and each result is ready a random latency after its exposure, in order.
So exposure times are regular and the step length (t_k+1 - t_k) is what jitters, not the other way round.

BatchSim does not reset finished envs by itself: call reset(done) after step().
"""

import dataclasses

import numpy as np

from sim import obs as obs_mod
from sim.config import KIND_PARAMS, SimConfig
from sim.trajectories import Trajectories

MIN_RESULT_GAP_S = 1e-3     # results come out of the pipeline one at a time: at least this far apart


class BatchSim:

    def __init__(self, n, cfg = SimConfig(), seed = None):
        self.n, self.cfg = n, cfg
        self.rng = np.random.default_rng(seed)
        self.difficulty = cfg.difficulty
        self.W, self.H = cfg.frame_size
        self.frame_dt = 1.0 / cfg.fps
        self.K = cfg.substeps
        self.M = cfg.history_frames * cfg.substeps
        self.tracker_cfg = dataclasses.replace(cfg.tracker, frame_size = cfg.frame_size)
        self.param_defs = cfg.params()
        s = cfg.servo
        self.lim_lo = np.array([s.pan_limits[0], s.tilt_limits[0]])
        self.lim_hi = np.array([s.pan_limits[1], s.tilt_limits[1]])
        self.iso = np.array([1.0, self.H / self.W])      # e (half width, half height) -> half widths on both axes

        z1, z2 = lambda: np.zeros(n), lambda: np.zeros((n, 2))
        self.p = {k: z1() for k in (*self.param_defs, *KIND_PARAMS)}    # this episode's parameters per env
        self.kind = np.zeros(n, int)                     # index into cfg.target.kinds
        self.t, self.steps, self.lost_s = z1(), np.zeros(n, int), z1()
        self.t_exp = z1()                                # exposure time of the current observation
        self.cmd, self.pending, self.pending_t, self.cmd_eff = z2(), z2(), z1(), z2()
        self.latched, self.next_pwm, self.motor, self.out = z2(), z1(), z2(), z2()
        self.tgt = z2()
        self.a_hist = np.zeros((n, cfg.action_history, 2))   # last actions, newest first
        self.occ_until, self.next_occ = z1(), z1()
        self.h_t = np.zeros((n, self.M))
        self.h_cam, self.h_tgt = np.zeros((n, self.M, 2)), np.zeros((n, self.M, 2))
        self.h_occ = np.zeros((n, self.M), bool)
        self.ring = 0
        self.traj = Trajectories(n, self.rng)
        self.est = obs_mod.BatchEstimator(n, self.tracker_cfg)
        self.obs = np.zeros((n, len(obs_mod.obs_names(cfg.action_history))), np.float32)
        self.last = dict(box = np.zeros((n, 4)), found = np.zeros(n, bool), latency = z1(),    # last observation,
                         e = z2(), valid = z1())                                                # for render / analysis
        self.reset()


    def seed(self, seed):
        self.rng = self.traj.rng = np.random.default_rng(seed)


    # ---------------------------------------------------------------- reset

    def reset(self, mask = None):
        """New episodes in the envs selected by mask (default: all). Returns all observations."""
        idx = np.arange(self.n) if mask is None else np.flatnonzero(mask)
        if len(idx) == 0:
            return self.obs
        k, rng, d, cfg = len(idx), self.rng, self.difficulty, self.cfg
        for name, par in self.param_defs.items():
            self.p[name][idx] = par.sample(rng, k, d)
        tp = cfg.target
        kind = rng.choice(len(tp.kinds), k, p = np.array(tp.kind_weights) / sum(tp.kind_weights))
        self.kind[idx] = kind
        family = np.zeros(k, int)
        for i, kd in enumerate(tp.kinds):               # per-kind size, speed, detection, motion
            m = kind == i
            for name in KIND_PARAMS:
                self.p[name][idx[m]] = getattr(kd, name).sample(rng, m.sum(), d)
            w = np.array(kd.family_weights, float)
            family[m] = rng.choice(len(w), m.sum(), p = w / w.sum())
        p = {name: v[idx] for name, v in self.p.items()}
        self.f_px = (self.W / 2) / np.tan(np.radians(self.p["hfov_deg"]) / 2)

        self.t[idx], self.steps[idx], self.lost_s[idx] = 0.0, 0, 0.0
        start = rng.uniform(self.lim_lo * 0.5, self.lim_hi * 0.5, (k, 2))
        for a in (self.cmd, self.pending, self.cmd_eff, self.latched, self.motor, self.out):
            a[idx] = start
        self.pending_t[idx] = 0.0
        self.next_pwm[idx] = rng.uniform(0, p["pwm_period_ms"] / 1e3)
        self.a_hist[idx] = 0.0

        half_fov = np.column_stack([p["hfov_deg"], p["hfov_deg"] * self.H / self.W]) / 2
        ang = rng.uniform(0, 2 * np.pi, k)              # random direction, exactly init_offset of the way to the edge
        off = np.column_stack([np.cos(ang), np.sin(ang)]) * p["init_offset"][:, None] * half_fov
        self.tgt[idx] = np.clip(start + off, self.lim_lo, self.lim_hi)
        self.traj.reset(idx, family, np.degrees(p["speed_mps"] / p["distance_m"]), p["distance_m"])
        self.occ_until[idx] = -1.0
        self.next_occ[idx] = self.next_occlusion(np.zeros(k), p["occlusion_rate_hz"])

        h0 = self.frame_dt / self.K                      # history: standing still before t = 0
        self.h_t[idx] = -h0 * ((self.ring - 1 - np.arange(self.M)) % self.M)
        self.h_cam[idx] = (start + np.column_stack([p["mount_pan_deg"], p["mount_tilt_deg"]]))[:, None]
        self.h_tgt[idx] = self.tgt[idx, None]
        self.h_occ[idx] = False

        self.est.reset(idx)
        self.t_exp[idx] = -self.latency(idx)
        self.obs[idx] = self.observe(idx)
        return self.obs


    def latency(self, idx):
        """Exposure (middle) -> result ready, sampled per frame for the envs idx, seconds"""
        return (self.p["latency_ms"][idx] + self.rng.exponential(1.0, len(idx)) * self.p["latency_tail_ms"][idx]) / 1e3


    def next_occlusion(self, t, rate):
        with np.errstate(divide = "ignore"):
            return t + np.where(rate > 0, self.rng.exponential(1.0, len(t)) / np.maximum(rate, 1e-9), np.inf)


    # ---------------------------------------------------------------- step

    def step(self, action):
        """action (N, 2) in [-1, 1]: commanded pan, tilt change of action * max_step_deg.
        Returns obs, reward, terminated, truncated, info (arrays over envs)."""
        cfg, p, rng = self.cfg, self.p, self.rng
        a = np.clip(np.asarray(action, np.float64).reshape(self.n, 2), -1, 1)
        self.cmd = np.clip(self.cmd + a * cfg.servo.max_step_deg, self.lim_lo, self.lim_hi)
        self.pending[:] = self.cmd
        self.pending_t = self.t + p["act_latency_ms"] / 1e3

        self.t_exp = self.t_exp + self.frame_dt * (1 + (rng.random(self.n) < p["skip_prob"]))    # a skipped frame
        t_ready = np.maximum(self.t_exp + self.latency(np.arange(self.n)), self.t + MIN_RESULT_GAP_S)  # in order
        dt = t_ready - self.t
        vel = self.traj.update(self.t, dt)
        start = self.t >= self.next_occ
        self.occ_until[start] = self.t[start] + p["occlusion_s"][start]
        self.next_occ[start] = self.next_occlusion(self.occ_until[start], p["occlusion_rate_hz"][start])

        h = dt / self.K
        q, period = p["quant_deg"][:, None], p["pwm_period_ms"] / 1e3
        tau, vmax = p["tau_s"][:, None], p["max_speed_dps"][:, None]
        dead, play = p["deadband_deg"][:, None], p["backlash_deg"][:, None] / 2
        mount = np.column_stack([p["mount_pan_deg"], p["mount_tilt_deg"]])
        for _ in range(self.K):
            t1 = self.t + h
            ready = t1 >= self.pending_t
            self.cmd_eff[ready] = self.pending[ready]
            latch = t1 >= self.next_pwm
            self.latched[latch] = (np.round(self.cmd_eff / q) * q)[latch]
            self.next_pwm[latch] += period[latch]
            err = self.latched - self.motor
            v = np.clip(err / tau, -vmax, vmax)
            v[np.abs(err) <= dead] = 0.0
            self.motor += v * h[:, None]
            np.clip(self.out, self.motor - play, self.motor + play, out = self.out)     # backlash
            self.tgt += vel * h[:, None]
            np.clip(self.tgt, self.lim_lo, self.lim_hi, out = self.tgt)
            self.t = t1
            j = self.ring
            self.h_t[:, j], self.h_cam[:, j], self.h_tgt[:, j] = self.t, self.out + mount, self.tgt
            self.h_occ[:, j] = self.t < self.occ_until
            self.ring = (j + 1) % self.M
        self.t = t_ready
        self.traj.bounce(self.tgt <= self.lim_lo, self.tgt >= self.lim_hi)

        a_prev = self.a_hist[:, 0].copy()
        self.a_hist = np.concatenate([a[:, None], self.a_hist[:, :-1]], axis = 1)
        self.obs = self.observe(np.arange(self.n))
        e_true, in_frame = self.true_error()
        self.lost_s = np.where(in_frame, 0.0, self.lost_s + dt)
        reward_step = self.reward(a, a_prev, e_true, in_frame)
        self.steps += 1
        terminated = self.lost_s > cfg.lost_terminate_s
        reward = reward_step - cfg.reward.fail_penalty * terminated
        truncated = ~terminated & (self.t >= cfg.episode_s)
        info = dict(e_true = e_true, in_frame = in_frame, t = self.t.copy(), privileged = self.privileged(e_true),
                    reward_step = reward_step)          # without the failure penalty (for the evaluation)
        return self.obs, reward, terminated, truncated, info


    # ---------------------------------------------------------------- camera and detector

    def project(self, cam, tgt, idx):
        """Target box in pixels for the envs idx, unclipped, and whether it is in front of the camera"""
        p = {k: v[idx] for k, v in self.p.items()}
        az, el = np.radians(tgt[:, 0]), np.radians(tgt[:, 1])
        pan, tilt = np.radians(cam[:, 0]), np.radians(cam[:, 1])
        ux, uy, uz = np.cos(el) * np.sin(az), np.sin(el), np.cos(el) * np.cos(az)
        x1 = ux * np.cos(pan) - uz * np.sin(pan)        # undo pan (about the vertical axis) ...
        z1 = ux * np.sin(pan) + uz * np.cos(pan)
        y2 = uy * np.cos(tilt) - z1 * np.sin(tilt)      # ... then tilt
        z2 = uy * np.sin(tilt) + z1 * np.cos(tilt)
        front = z2 > 1e-3
        zs = np.where(front, z2, 1.0)
        f = self.f_px[idx]
        u, v = self.W / 2 + f * x1 / zs, self.H / 2 - f * y2 / zs
        bw = f * p["width_m"] / (p["distance_m"] * zs)
        bh = bw * p["aspect"]
        return np.column_stack([u - bw / 2, v - bh / 2, u + bw / 2, v + bh / 2]), front


    def true_error(self):
        mount = np.column_stack([self.p["mount_pan_deg"], self.p["mount_tilt_deg"]])
        box, front = self.project(self.out + mount, self.tgt, np.arange(self.n))
        c = obs_mod.box_centres(box)
        e = np.column_stack([c[:, 0] - self.W / 2, self.H / 2 - c[:, 1]]) / np.array([self.W / 2, self.H / 2])
        in_frame = front & (np.abs(e) <= 1).all(axis = 1)
        return np.where(front[:, None], e, 1.0).clip(-1, 1), in_frame


    def detect(self, cam, tgt, occluded, idx, rel_vel = None):
        """The detector's box (clipped to the frame, with noise and motion blur) and whether it found the target.
        rel_vel (k, 2): target angular velocity relative to the camera during the exposure, deg/s"""
        p, rng, n, det = {k: v[idx] for k, v in self.p.items()}, self.rng, len(idx), self.cfg.detector
        box, front = self.project(cam, tgt, idx)
        c = obs_mod.box_centres(box)
        bw, bh = box[:, 2] - box[:, 0], box[:, 3] - box[:, 1]
        size = np.sqrt(np.abs(bw * bh))
        big_enough = np.minimum(bw, bh) >= p["min_box_px"]
        sigma = p["noise_px"] + p["noise_per_px"] * size
        c = c + rng.standard_normal((n, 2)) * sigma[:, None]
        scale = 1 + rng.standard_normal(n) * p["size_noise"]
        streak = np.zeros((n, 2)) if rel_vel is None else \
            np.abs(self.f_px[idx, None] * np.radians(rel_vel) * p["exposure_ms"][:, None] / 1e3)
        bw, bh = bw * scale + streak[:, 0], bh * scale + streak[:, 1]       # the box covers the whole streak
        blur_miss = np.clip(det.blur_miss * (np.linalg.norm(streak, axis = 1) / np.maximum(size, 1e-6) - 1),
                            0, det.blur_miss_max)
        fp = rng.random(n) < p["false_pos_prob"]          # a wrong box near the target
        c[fp] += rng.uniform(-1, 1, (fp.sum(), 2)) * self.tracker_cfg.gate * self.W
        full = np.column_stack([c[:, 0] - bw / 2, c[:, 1] - bh / 2, c[:, 0] + bw / 2, c[:, 1] + bh / 2])
        out = full.copy()
        out[:, [0, 2]] = out[:, [0, 2]].clip(0, self.W)
        out[:, [1, 3]] = out[:, [1, 3]].clip(0, self.H)
        area = lambda b: (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        visible = area(out) / np.maximum(area(full), 1e-9)
        found = (front & ~occluded & (visible >= det.min_visible) & big_enough
                 & (rng.random(n) >= p["drop_prob"]) & (rng.random(n) >= blur_miss))
        return out, found


    def observe(self, idx):
        """Observations of the envs idx: the scene at the exposure time t_exp, through the detector and the
        tracker. The scene comes from the nearest physics substep; the tracker gets the exact exposure time,
        as the Pi gets the sensor timestamp."""
        t_seen = self.t_exp[idx]
        j = np.abs(self.h_t[idx] - t_seen[:, None]).argmin(axis = 1)
        jp = (j - 1) % self.M                           # the substep before (never the oldest: latency << history)
        rel = lambda q: self.h_tgt[idx, q] - self.h_cam[idx, q]
        rel_vel = (rel(j) - rel(jp)) / np.maximum(self.h_t[idx, j] - self.h_t[idx, jp], 1e-6)[:, None]
        box, found = self.detect(self.h_cam[idx, j], self.h_tgt[idx, j], self.h_occ[idx, j], idx, rel_vel)
        e, de, size, valid, t_lost = self.est.update(box, found, t_seen, idx)
        for k, v in dict(box = box, found = found, latency = self.t[idx] - t_seen, e = e, valid = valid).items():
            self.last[k][idx] = v
        s = self.cfg.servo
        return obs_mod.build(e, de, size, valid, t_lost, self.a_hist[idx], self.cmd[idx], s.pan_limits, s.tilt_limits)


    # ---------------------------------------------------------------- reward and critic state

    def reward(self, a, a_prev, e_true, in_frame):
        w = self.cfg.reward
        e2 = ((e_true * self.iso) ** 2).sum(axis = 1)     # both axes in half frame widths
        return (- w.w_e * e2 - w.w_s * ((a - a_prev) ** 2).sum(axis = 1) - w.w_a * (a ** 2).sum(axis = 1)
                + w.w_c * (in_frame & (np.sqrt(e2) < w.eps)) - w.w_l * ~in_frame)


    def privileged(self, e_true):
        """Extra critic input (PROJECT.md 5.4): true state and this episode's parameters, scaled to ~[-1, 1]"""
        half = (self.lim_hi - self.lim_lo) / 2
        mid = (self.lim_hi + self.lim_lo) / 2
        tp = self.cfg.target
        defs = {**self.param_defs, **{k: tp.kind_param_range(k) for k in KIND_PARAMS}}
        params = [(self.p[k] - d.lo) / max(d.hi - d.lo, 1e-9) * 2 - 1 for k, d in defs.items()]
        kind = np.eye(len(tp.kinds))[self.kind]          # the critic may know what it is tracking; the actor doesn't
        return np.column_stack([e_true, self.traj.vel / 180.0, (self.out - mid) / half,
                                (self.latched - self.out) / 10.0, self.last["latency"][:, None] / 0.05, *params, kind]
                               ).astype(np.float32)
