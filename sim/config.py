"""Simulator parameters: nominal values and per-episode randomization ranges (PROJECT.md 4 and 5.1).

Units are degrees, seconds, metres and stream pixels. Camera, latency and detector values come from
the measurements so far (perception/bench/plan.md, track.csv). Servo values are MG90S estimates
until system identification (Phase 4) replaces them.

`difficulty` (0..1) is the curriculum knob: each parameter is sampled from
[nominal + d * (lo - nominal), nominal + d * (hi - nominal)], so 0 gives the nominal values only
and 1 the full randomization ranges.
"""

from dataclasses import dataclass, field, fields


@dataclass(frozen=True)
class Param:
    nominal: float
    lo: float
    hi: float

    def sample(self, rng, n, difficulty):
        lo = self.nominal + difficulty * (self.lo - self.nominal)
        hi = self.nominal + difficulty * (self.hi - self.nominal)
        return rng.uniform(lo, hi, n)




@dataclass(frozen=True)
class CameraParams:
    hfov_deg: Param = Param(48.8, 46.0, 51.5)           # datasheet value, not calibrated yet
    mount_pan_deg: Param = Param(0.0, -1.5, 1.5)        # optical axis vs gimbal angle
    mount_tilt_deg: Param = Param(0.0, -1.5, 1.5)
    latency_ms: Param = Param(37.0, 30.0, 55.0)         # middle of exposure -> observation, base
    latency_tail_ms: Param = Param(1.5, 0.5, 6.0)       # plus an exponential tail with this mean
    skip_prob: Param = Param(0.003, 0.0, 0.03)          # a frame is skipped (the step takes 2 frames)




@dataclass(frozen=True)
class DetectorParams:
    noise_px: Param = Param(1.0, 0.3, 3.0)              # box centre jitter, 1 sigma ...
    noise_per_px: Param = Param(0.006, 0.0, 0.02)       # ... plus this times the box size (sqrt of area)
    size_noise: Param = Param(0.02, 0.005, 0.06)        # relative box size jitter
    drop_prob: Param = Param(0.003, 0.0, 0.08)          # missed detection per frame
    false_pos_prob: Param = Param(0.0, 0.0, 0.01)       # box jumps to a wrong spot near the target
    min_box_px: Param = Param(8.0, 6.0, 14.0)           # smaller boxes are not detected
    min_visible: float = 0.25                           # share of the box that must be inside the frame




@dataclass(frozen=True)
class ServoParams:
    max_speed_dps: Param = Param(400.0, 200.0, 700.0)   # slew rate limit
    tau_s: Param = Param(0.03, 0.01, 0.08)              # first-order lag time constant
    deadband_deg: Param = Param(0.5, 0.1, 1.2)
    backlash_deg: Param = Param(1.0, 0.2, 2.5)
    quant_deg: Param = Param(0.44, 0.1, 0.9)            # PCA9685 12 bit at 50 Hz: ~0.44 deg per count
    pwm_period_ms: Param = Param(20.0, 20.0, 20.0)      # the servo takes a new command once per PWM frame
    act_latency_ms: Param = Param(3.0, 1.0, 10.0)       # policy + I2C until the driver has the command
    pan_limits: tuple = (-90.0, 90.0)
    tilt_limits: tuple = (-45.0, 45.0)
    max_step_deg: float = 3.0                           # action 1.0 = this commanded change per step




@dataclass(frozen=True)
class TargetParams:
    distance_m: Param = Param(3.0, 1.0, 8.0)
    width_m: Param = Param(0.4, 0.08, 0.7)
    aspect: Param = Param(2.0, 0.5, 4.0)                # box height / width
    speed_dps: Param = Param(10.0, 2.0, 60.0)           # typical angular speed of the trajectory
    occlusion_rate_hz: Param = Param(0.0, 0.0, 0.3)     # occlusions per second
    occlusion_s: Param = Param(0.3, 0.05, 1.5)          # occlusion length
    init_offset: Param = Param(0.3, 0.0, 0.9)           # start offset from the frame centre, share of half the FOV
    # still, random walk, sinusoid, direction changes, bursts, stop-and-go (see trajectories.py)
    family_weights: tuple = (0.1, 0.25, 0.2, 0.2, 0.1, 0.15)




@dataclass(frozen=True)
class RewardParams:
    w_e: float = 1.0            # centering: - w_e * |e|^2
    w_s: float = 0.2            # smoothness: - w_s * |a - a_prev|^2
    w_a: float = 0.01           # effort: - w_a * |a|^2
    w_c: float = 0.1            # bonus when |e| < eps
    eps: float = 0.05
    w_l: float = 1.0            # per step with the target outside the frame




@dataclass(frozen=True)
class SimConfig:
    camera: CameraParams = field(default_factory = CameraParams)
    detector: DetectorParams = field(default_factory = DetectorParams)
    servo: ServoParams = field(default_factory = ServoParams)
    target: TargetParams = field(default_factory = TargetParams)
    reward: RewardParams = field(default_factory = RewardParams)
    frame_size: tuple = (640, 360)
    fps: float = 60.0               # one step per camera frame result, like on the Pi
    episode_s: float = 20.0         # truncate after this long
    lost_terminate_s: float = 1.5   # terminate after the target is outside the frame this long
    difficulty: float = 1.0
    substeps: int = 10              # physics steps per frame
    history_frames: int = 8         # how far back observations can look (covers the latency)

    def params(self):
        """All randomized parameters as {name: Param}"""
        out = {}
        for section in (self.camera, self.detector, self.servo, self.target):
            for f in fields(section):
                v = getattr(section, f.name)
                if isinstance(v, Param):
                    out[f.name] = v
        return out
