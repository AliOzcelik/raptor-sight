"""PID baseline controller (PROJECT.md 9.2): per axis, on the tracker's error, for sim and Pi alike.

Input is the policy observation (sim/obs.py), output the same action as the RL policy: the commanded
pan/tilt change per step as a share of max_step_deg. It only uses what the Pi knows (nominal FOV and
frame rate), not the simulator's randomized values.

  err  = tracker error as an angle (deg)
  derr = tracker error velocity as an angle rate (deg/s): already filtered by the tracker
  u    = kp * err + ki * integral(err) + kd * derr          (deg this step)

Anti-windup: the integral only grows on real detections, never further into saturation, and its
term is limited to i_max_deg. Lost target: follow the tracker's prediction (P only) for hold_s,
then hold still. The prediction is in the image of the last detection and doesn't know the camera
has turned since, so the turn commanded since the loss is subtracted from it (otherwise the same
stale error is corrected again every step and the camera runs away).

Gains can be scalars or one value per env (used by tune_pid.py to try many gains at once).
"""

from dataclasses import dataclass

import numpy as np

from sim.config import ServoParams
from sim.obs import BASE_NAMES, DE_SCALE

IDX = {name: k for k, name in enumerate(BASE_NAMES)}


@dataclass(frozen=True)
class PIDGains:
    kp: float = 0.3         # share of the error angle corrected per step
    ki: float = 0.0         # 1/s
    kd: float = 0.0         # s: error rate (deg/s) x kd = deg per step
    i_max_deg: float = 1.5  # limit of the integral term, deg per step
    hold_s: float = 0.3     # follow the tracker's prediction this long after losing the target, then hold




class PID:

    def __init__(self, n, gains = PIDGains(), hfov_deg = 48.8, frame_size = (640, 360), fps = 60.0,
                 max_step_deg = ServoParams().max_step_deg, t_lost_max_s = 1.0):
        col = lambda v: np.broadcast_to(np.asarray(v, np.float64), (n,))[:, None].copy()
        self.kp, self.ki, self.kd = col(gains.kp), col(gains.ki), col(gains.kd)
        self.i_max, self.hold_s = col(gains.i_max_deg), col(gains.hold_s)[:, 0]
        t = np.tan(np.radians(hfov_deg / 2))
        self.tan_half = np.array([t, t * frame_size[1] / frame_size[0]])     # pinhole: e = 1 at the frame edge
        self.dt, self.max_step, self.t_lost_max = 1.0 / fps, max_step_deg, t_lost_max_s
        self.integral = np.zeros((n, 2))
        self.moved = np.zeros((n, 2))       # deg commanded since the last detection


    def reset(self, mask = None):
        m = slice(None) if mask is None else mask
        self.integral[m] = 0.0
        self.moved[m] = 0.0


    def __call__(self, obs):
        e = obs[:, [IDX["e_x"], IDX["e_y"]]].astype(np.float64)
        de = obs[:, [IDX["de_x"], IDX["de_y"]]].astype(np.float64) / DE_SCALE
        valid = obs[:, IDX["valid"]] > 0.5
        lost_s = obs[:, IDX["t_lost"]] * self.t_lost_max

        x = e * self.tan_half
        err = np.degrees(np.arctan(x))
        derr = np.degrees(de * self.tan_half / (1 + x ** 2))
        u = self.kp * err + self.ki * self.integral + self.kd * derr

        saturated = (np.abs(u) >= self.max_step) & (np.sign(err) == np.sign(u))
        grow = valid[:, None] & ~saturated
        limit = self.i_max / np.maximum(self.ki, 1e-9)
        self.integral = np.where(grow, np.clip(self.integral + err * self.dt, -limit, limit), self.integral)

        follow = ~valid & (lost_s < self.hold_s)        # P on the tracker's prediction, minus the turn since
        u = np.where(valid[:, None], u, np.where(follow[:, None], self.kp * (err - self.moved), 0.0))
        a = np.clip(u / self.max_step, -1, 1)
        self.moved = np.where(valid[:, None], 0.0, self.moved + a * self.max_step)
        return a
