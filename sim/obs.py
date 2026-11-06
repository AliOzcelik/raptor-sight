"""Tracker and observation builder shared by the simulator and the Pi.

BatchEstimator is a vectorized copy of perception/target.py (TargetSelector + StateEstimator) for
one target per env, so the policy sees the same filtering in sim as on the Pi; sim/check_sim.py
checks that both give the same numbers. build() makes the policy input of PROJECT.md 3.1; the
deploy loop calls from_tracker() with the same constants.

The policy input ends with the last K actions, newest first (K = SimConfig.action_history). A command
takes ~5 frames to show in the measured error (latency + servo), so the frame a policy sees doesn't yet
show the turns it has already commanded; the history tells it what is still on the way.
"""

import numpy as np

BASE_NAMES = ("e_x", "e_y", "de_x", "de_y", "w", "h", "pan_cmd", "tilt_cmd", "valid", "t_lost")
DE_SCALE = 0.25         # error velocity 4 half-frames/s -> 1


def obs_names(action_history):
    """Feature names: BASE_NAMES (fixed positions), then a1_x, a1_y (the last action), a2_x, ..."""
    return BASE_NAMES + tuple(f"a{j}_{ax}" for j in range(1, action_history + 1) for ax in "xy")


def build(e, de, size, valid, t_lost, a_hist, cmd, pan_limits, tilt_limits):
    """(N, 10 + 2K) float32 policy input, all roughly in [-1, 1]. a_hist (N, K, 2) the last K actions,
    newest first; cmd = commanded pan, tilt in degrees"""
    lim = np.array([pan_limits, tilt_limits], np.float64)
    mid, half = lim.mean(axis = 1), (lim[:, 1] - lim[:, 0]) / 2
    return np.column_stack([e, np.clip(de * DE_SCALE, -1, 1), size, (cmd - mid) / half,
                            valid, t_lost, a_hist.reshape(len(a_hist), -1)]).astype(np.float32)


def from_tracker(obs, a_hist, cmd, pan_limits, tilt_limits):
    """Policy input from a perception.target.Observation on the Pi. a_hist (K, 2): the last K actions sent,
    newest first (zeros before the first ones)"""
    a_hist = np.asarray(a_hist, np.float64)
    return build(np.array([[obs.e_x, obs.e_y]]), np.array([[obs.de_x, obs.de_y]]),
                 np.array([[obs.w, obs.h]]), np.array([float(obs.valid)]), np.array([obs.t_lost]),
                 a_hist[None], np.asarray(cmd, np.float64).reshape(1, 2), pan_limits, tilt_limits)[0]




def box_iou(a, b):
    ix = (np.minimum(a[:, 2], b[:, 2]) - np.maximum(a[:, 0], b[:, 0])).clip(0)
    iy = (np.minimum(a[:, 3], b[:, 3]) - np.maximum(a[:, 1], b[:, 1])).clip(0)
    inter = ix * iy
    area = lambda r: (r[:, 2] - r[:, 0]) * (r[:, 3] - r[:, 1])
    return inter / np.maximum(area(a) + area(b) - inter, 1e-6)


def box_centres(b):
    return np.column_stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2])




class BatchEstimator:
    """perception/target.py for N envs with one target each. Times in seconds."""

    def __init__(self, n, cfg):
        self.cfg = cfg
        self.half = np.array(cfg.frame_size, np.float64) / 2
        self.box = np.zeros((n, 4))             # selector: last matched box
        self.locked = np.zeros(n, bool)
        self.t_seen = np.zeros(n)
        self.pos = np.zeros((n, 2))             # estimator: filtered e_x, e_y at t_match
        self.vel = np.zeros((n, 2))
        self.size = np.zeros((n, 2))
        self.has_pos = np.zeros(n, bool)
        self.t_match = np.zeros(n)
        self.pend_box = np.zeros((n, 4))        # a jump waiting for confirmation (TargetSelector.pending)
        self.pend_t = np.zeros(n)
        self.has_pend = np.zeros(n, bool)


    def reset(self, idx):
        self.locked[idx] = False
        self.has_pos[idx] = False
        self.has_pend[idx] = False
        self.vel[idx] = 0.0
        self.size[idx] = 0.0


    STATE = ("box", "locked", "t_seen", "pos", "vel", "size", "has_pos", "t_match", "pend_box", "pend_t", "has_pend")

    def same_object(self, a, b):
        cfg = self.cfg
        return ((box_iou(a, b) >= cfg.min_iou)
                | (np.linalg.norm(box_centres(b) - box_centres(a), axis = 1) <= cfg.gate * cfg.frame_size[0]))


    def update(self, box, detected, t, idx):
        """Envs idx: box (k, 4) x1, y1, x2, y2 in pixels, detected (k,), t (k,) exposure time.
        Returns e, de, size (k, 2) and valid, t_lost (k,) like target.Observation."""
        cfg = self.cfg
        s = {k: getattr(self, k)[idx] for k in self.STATE}      # copies; written back at the end
        expired = t - s["t_seen"] > cfg.lose_after_s            # lost for too long
        s["locked"] &= ~expired
        s["has_pend"] &= ~expired & ~(t - s["pend_t"] > cfg.confirm_s)
        new = detected & ~s["locked"]
        pos, vel = s["pos"], s["vel"]
        since = t - s["t_match"]
        pred = pos + vel * since[:, None]
        pc = np.column_stack([self.half[0] * (1 + pred[:, 0]), self.half[1] * (1 - pred[:, 1])])
        ref_pred = s["box"] + np.tile(pc - box_centres(s["box"]), 2)     # last box moved to the prediction
        same_pend = s["has_pend"] & self.same_object(s["pend_box"], box)
        cand = new | (detected & s["locked"] & ((s["has_pos"] & self.same_object(ref_pred, box))
                                                | self.same_object(s["box"], box) | same_pend))

        c = box_centres(box)
        z = np.column_stack([c[:, 0] - self.half[0], self.half[1] - c[:, 1]]) / self.half    # + right, + up
        limit = cfg.jump * cfg.frame_size[0] + cfg.jump_speed * np.linalg.norm(vel * self.half, axis = 1) * since
        jump = cand & ~new & s["has_pos"] & (np.linalg.norm((z - pred) * self.half, axis = 1) > limit)
        restart = jump & same_pend                              # two boxes agree: the target really moved
        hold = jump & ~restart                                  # held back; this frame counts as missed
        s["pend_box"][hold], s["pend_t"][hold], s["has_pend"][hold] = box[hold], t[hold], True
        matched = cand & ~hold
        s["box"][matched], s["t_seen"][matched], s["locked"][matched] = box[matched], t[matched], True
        s["has_pend"][matched] = False

        start = matched & (new | ~s["has_pos"])
        pos[start], vel[start] = z[start], 0.0
        pc = box_centres(s["pend_box"][restart])                # the jump this box confirms (still the old one)
        zp = np.column_stack([pc[:, 0] - self.half[0], self.half[1] - pc[:, 1]]) / self.half
        vel[restart] = (z[restart] - zp) / np.maximum(t[restart] - s["pend_t"][restart], 1e-3)[:, None]
        pos[restart] = z[restart]
        go = matched & ~start & ~restart
        dt = np.maximum(t - s["t_match"], 1e-3)[go, None]
        pred = pos[go] + vel[go] * dt
        r = z[go] - pred
        pos[go] = pred + cfg.alpha * r
        vel[go] = vel[go] + cfg.beta * r / dt
        s["size"][matched] = np.column_stack([box[matched, 2] - box[matched, 0],
                                              box[matched, 3] - box[matched, 1]]) / (2 * self.half)
        s["t_match"][matched], s["has_pos"][matched] = t[matched], True

        lost = ~matched
        e = z.copy()
        e[lost] = np.where(s["has_pos"][lost, None],
                           (pos[lost] + vel[lost] * (t - s["t_match"])[lost, None]).clip(-1, 1), 0.0)
        t_lost = np.where(matched, 0.0,
                          np.where(s["has_pos"], np.minimum(1.0, (t - s["t_match"]) / cfg.t_lost_max_s), 1.0))
        for k in self.STATE:
            getattr(self, k)[idx] = s[k]
        return e, vel.copy(), s["size"].copy(), matched.astype(np.float64), t_lost
