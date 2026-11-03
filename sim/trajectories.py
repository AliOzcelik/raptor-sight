"""Target motion in world angles (azimuth + right, elevation + up, degrees), vectorized over envs.

Families (PROJECT.md 4.1). Each step sets a velocity; the simulator moves the target with it
during the step and keeps it inside the reachable range.
"""

import numpy as np

FAMILIES = ("still", "random_walk", "sinusoid", "direction_changes", "bursts", "stop_and_go")
STILL, WALK, SINE, TURNS, BURSTS, STOPGO = range(len(FAMILIES))
AXIS_SCALE = np.array([1.0, 0.5])      # targets move less vertically than horizontally
WALK_TAU_S = 1.0                        # velocity correlation time of the random walk


class Trajectories:

    def __init__(self, n, rng):
        self.rng = rng
        self.family = np.zeros(n, int)
        self.speed = np.zeros(n)                # typical angular speed, deg/s
        self.vel = np.zeros((n, 2))             # deg/s
        self.next_event = np.zeros(n)           # next direction change / burst / stop-go toggle
        self.burst_until = np.zeros(n)
        self.moving = np.zeros(n, bool)
        self.omega = np.zeros((n, 2))           # sinusoid, rad/s
        self.phase = np.zeros((n, 2))


    def reset(self, idx, family, speed):
        k, rng = len(idx), self.rng
        self.family[idx] = family
        self.speed[idx] = np.where(family == STILL, 0.0, speed)
        self.vel[idx] = 0.0
        self.next_event[idx] = rng.uniform(0.3, 1.5, k)
        self.burst_until[idx] = 0.0
        self.moving[idx] = False
        self.omega[idx] = 2 * np.pi * rng.uniform(0.1, 0.6, (k, 2))
        self.phase[idx] = rng.uniform(0, 2 * np.pi, (k, 2))


    def directions(self, m):
        a = self.rng.uniform(0, 2 * np.pi, m.sum())
        return np.column_stack([np.cos(a), np.sin(a)])


    def update(self, t, dt):
        """Velocity for the step starting at t (per env) and lasting dt"""
        rng, f = self.rng, self.family
        s = self.speed[:, None] * AXIS_SCALE
        event = t >= self.next_event

        def walk(m, scale):
            d = dt[m, None] / WALK_TAU_S
            self.vel[m] += -self.vel[m] * d + scale * s[m] * np.sqrt(2 * d) * rng.standard_normal((m.sum(), 2))

        walk(f == WALK, 1.0)

        m = f == SINE                           # peak speed = speed
        self.vel[m] = s[m] * np.cos(self.omega[m] * t[m, None] + self.phase[m])

        m = (f == TURNS) & event                # new random velocity at each event
        self.vel[m] = s[m] * self.directions(m) * rng.uniform(0.5, 1.5, (m.sum(), 1))
        self.next_event[m] = t[m] + rng.uniform(0.5, 3.0, m.sum())

        m = (f == BURSTS) & event               # short fast bursts out of a slow walk
        self.vel[m] = 3 * s[m] * self.directions(m)
        self.burst_until[m] = t[m] + rng.uniform(0.2, 0.6, m.sum())
        self.next_event[m] = self.burst_until[m] + rng.uniform(1.0, 4.0, m.sum())
        walk((f == BURSTS) & (t >= self.burst_until), 0.3)

        m = (f == STOPGO) & event               # alternate moving and standing
        self.moving[m] = ~self.moving[m]
        go = m & self.moving
        self.vel[go] = s[go] * self.directions(go)
        self.vel[m & ~self.moving] = 0.0
        self.next_event[m] = t[m] + rng.uniform(0.5, 2.0, m.sum())
        return self.vel


    def bounce(self, at_lo, at_hi):
        """Turn the velocity around on axes where the target hit the edge of its range"""
        out = (at_lo & (self.vel < 0)) | (at_hi & (self.vel > 0))
        self.vel[out] *= -1
