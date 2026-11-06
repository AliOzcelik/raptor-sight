from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class TrackerConfig:
    frame_size: tuple = (640, 360)      # stream size the boxes are in
    min_iou: float = 0.3                # follow the same-class box that overlaps the last one this much
    gate: float = 0.15                  # otherwise the nearest one within this share of the frame width
    lose_after_s: float = 0.5           # drop the lock after this long without a match
    t_lost_max_s: float = 1.0           # t_lost reaches 1 after this long
    alpha: float = 0.8                  # alpha-beta filter gains: position
    beta: float = 0.5                   # velocity (fast enough for kicked and thrown balls; tested in the sim)
    jump: float = 0.05                  # a box this share of the frame width off the prediction is a jump:
                                        # held back until a second box confirms it (one wrong box can't take the lock)
    jump_speed: float = 2.0             # ... plus this many times the predicted motion since the last match
                                        # (a ball that bounces back lands 2 v dt off the prediction)
    confirm_s: float = 0.1              # the confirming box must come within this long
    classes: tuple = ()                 # COCO class ids to track (e.g. (0,) person, (32,) sports ball); () = any




@dataclass
class Observation:
    """Target state for the controller. Errors are + right, + up, like RaspberryPiCamera.angles"""
    e_x: float = 0.0        # centre error / half the frame width, -1..1
    e_y: float = 0.0        # centre error / half the frame height, -1..1
    de_x: float = 0.0       # error velocity per second (filtered)
    de_y: float = 0.0
    w: float = 0.0          # box size / frame size, 0..1
    h: float = 0.0
    valid: bool = False     # target matched in this frame
    t_lost: float = 1.0     # time since the last match / t_lost_max_s, 0..1
    class_id: int = -1
    frame_id: int = -1
    t_exposure_ns: int = 0  # middle of the exposure of the frame this describes

    def as_array(self):
        return np.array([self.e_x, self.e_y, self.de_x, self.de_y, self.w, self.h,
                         float(self.valid), self.t_lost], np.float32)




def iou(box, boxes):
    """IoU of one box with each row of boxes"""
    ix = (np.minimum(box[2], boxes[:, 2]) - np.maximum(box[0], boxes[:, 0])).clip(0)
    iy = (np.minimum(box[3], boxes[:, 3]) - np.maximum(box[1], boxes[:, 1])).clip(0)
    inter = ix * iy
    area = lambda b: (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    return inter / np.maximum(area(box) + area(boxes) - inter, 1e-6)


def centres(boxes):
    return np.column_stack([(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2])




class TargetSelector:
    """Picks one box as the target and keeps following the same object"""

    def __init__(self, cfg):
        self.cfg = cfg
        self.centre = np.array(cfg.frame_size, np.float32) / 2
        self.box = None         # target box in the last frame it was accepted
        self.t_seen_ns = None
        self.new = False
        self.pending = None     # (box, t_ns) of a jump waiting for a second box to confirm it


    def lock(self, box):
        """Follow this box (e.g. chosen with a click) from the next frame on"""
        self.box, self.t_seen_ns, self.new, self.pending = np.asarray(box, np.float32), None, True, None


    def reset(self):
        """Forget the target; the next frame picks a new one"""
        self.box, self.pending = None, None


    def select(self, boxes, t_ns, predicted = None):
        """The candidate target box in this frame (or None) and whether it is a new target. Matches against
        `predicted` (the last box moved to where the estimator expects it: fast targets), else the last
        accepted box (a target that turned), else a pending jump. Call accept() if it is taken."""
        if self.cfg.classes:
            boxes = boxes[np.isin(boxes[:, 5], self.cfg.classes)]
        if self.box is not None and self.t_seen_ns is not None \
                and t_ns - self.t_seen_ns > self.cfg.lose_after_s * 1e9:
            self.box = self.pending = None          # lost for too long
        if self.pending is not None and t_ns - self.pending[1] > self.cfg.confirm_s * 1e9:
            self.pending = None
        if self.box is None:
            box = self.pick(boxes)
            self.new = box is not None
            return box, self.new
        box = None if predicted is None else self.match(boxes, predicted)
        if box is None:
            box = self.match(boxes, self.box)
        if box is None and self.pending is not None:
            box = self.match(boxes, self.pending[0])
        return box, self.new and box is not None


    def accept(self, box, t_ns):
        self.box, self.t_seen_ns, self.new, self.pending = box, t_ns, False, None


    def pick(self, boxes):
        """The box closest to the frame centre"""
        if len(boxes) == 0:
            return None
        return boxes[np.linalg.norm(centres(boxes) - self.centre, axis = 1).argmin()]


    def match(self, boxes, ref):
        """Same-class box that overlaps ref most, else the nearest one inside the gate"""
        same = boxes[boxes[:, 5] == ref[5]]
        if len(same) == 0:
            return None
        overlap = iou(ref, same)
        if overlap.max() >= self.cfg.min_iou:
            return same[overlap.argmax()]
        dist = np.linalg.norm(centres(same) - centres(ref[None])[0], axis = 1)
        if dist.min() <= self.cfg.gate * self.cfg.frame_size[0]:
            return same[dist.argmin()]
        return None


    def same_object(self, a, b):
        """Whether box b continues box a (the matching rule of match())"""
        return bool(iou(a, b[None])[0] >= self.cfg.min_iou or
                    np.linalg.norm(centres(a[None])[0] - centres(b[None])[0]) <= self.cfg.gate * self.cfg.frame_size[0])




class StateEstimator:
    """Alpha-beta filter on the centre error. The reported position is the measurement itself
    (no added lag); the filter gives the velocity and the prediction while the target is lost."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.half = np.array(cfg.frame_size, np.float32) / 2
        self.pos = None                             # filtered e_x, e_y at t_ns
        self.vel = np.zeros(2, np.float32)
        self.size = np.zeros(2, np.float32)
        self.class_id = -1
        self.t_ns = None                            # time of the last match


    def measure(self, box):
        c = centres(box[None])[0]
        return np.array([c[0] - self.half[0], self.half[1] - c[1]]) / self.half     # + right, + up


    def predict(self, t_ns):
        return self.pos + self.vel * ((t_ns - self.t_ns) / 1e9)


    def predicted_box(self, box, t_ns):
        """box moved so its centre is where the target is expected at t_ns"""
        p = self.predict(t_ns)
        c = np.array([self.half[0] * (1 + p[0]), self.half[1] * (1 - p[1])]) - centres(box[None])[0]
        out = box.copy()
        out[[0, 2]] += c[0]
        out[[1, 3]] += c[1]
        return out


    def is_jump(self, box, t_ns):
        """Whether box lies further off the predicted position than cfg.jump plus cfg.jump_speed times
        the predicted motion since the last match"""
        if self.pos is None:
            return False
        dt = (t_ns - self.t_ns) / 1e9
        limit = self.cfg.jump * self.cfg.frame_size[0] + self.cfg.jump_speed * np.linalg.norm(self.vel * self.half) * dt
        return bool(np.linalg.norm((self.measure(box) - self.predict(t_ns)) * self.half) > limit)


    def update(self, box, new, t_ns, restart = None):
        """restart = (box, t_ns) of a confirmed jump: start again from it and box, velocity from the two"""
        if box is not None:
            z = self.measure(box)
            if new or self.pos is None:
                self.pos, self.vel = z, np.zeros(2, np.float32)
            elif restart is not None:
                dt = max((t_ns - restart[1]) / 1e9, 1e-3)
                self.pos, self.vel = z, (z - self.measure(restart[0])) / dt
            else:
                dt = max((t_ns - self.t_ns) / 1e9, 1e-3)
                pred = self.pos + self.vel * dt
                r = z - pred
                self.pos = pred + self.cfg.alpha * r
                self.vel = self.vel + self.cfg.beta * r / dt
            self.size = np.array([box[2] - box[0], box[3] - box[1]]) / (2 * self.half)
            self.class_id, self.t_ns = int(box[5]), t_ns
            return self.observation(z, True, 0.0)

        if self.pos is None:                        # nothing seen yet
            return Observation()
        dt = (t_ns - self.t_ns) / 1e9
        pos = (self.pos + self.vel * dt).clip(-1, 1)    # where it should be by now
        return self.observation(pos, False, min(1.0, dt / self.cfg.t_lost_max_s))


    def observation(self, pos, valid, t_lost):
        return Observation(float(pos[0]), float(pos[1]), float(self.vel[0]), float(self.vel[1]),
                           float(self.size[0]), float(self.size[1]), valid, t_lost, self.class_id)




class Tracker:
    """Detections -> Observation, once per frame"""

    def __init__(self, cfg = TrackerConfig()):
        self.selector = TargetSelector(cfg)
        self.estimator = StateEstimator(cfg)


    def update(self, dets):
        t = dets.frame.t_exposure_ns                # when the boxes were seen, not when they were ready
        sel, est = self.selector, self.estimator
        predicted = est.predicted_box(sel.box, t) if sel.box is not None and not sel.new and est.pos is not None else None
        box, new = sel.select(dets.boxes, t, predicted)
        restart = None
        if box is not None and not new and self.estimator.is_jump(box, t):
            if sel.pending is not None and sel.same_object(sel.pending[0], box):
                restart = sel.pending               # two boxes agree: the target really moved
            else:
                sel.pending, box = (box, t), None   # hold it back; this frame counts as missed
        if box is not None:
            sel.accept(box, t)
        obs = self.estimator.update(box, new, t, restart)
        obs.frame_id, obs.t_exposure_ns = dets.frame.frame_id, t
        return obs
