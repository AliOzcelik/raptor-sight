"""Target motion in world angles (azimuth + right, elevation + up, degrees), vectorized over envs.

Families (PROJECT.md 4.1). Each step sets a velocity; the simulator moves the target with it
during the step and keeps it inside the reachable range. People use the first six, balls still,
direction changes (kicks), roll and throw (sim.config.PERSON / BALL).

  roll    kicked along the floor in a new direction every 1-4 s, slowing down by friction
  throw   thrown up and across, falls under gravity (9.81 m/s^2 / distance), bounces, comes to rest,
          and is thrown again
"""

import numpy as np

FAMILIES = ("still", "random_walk", "sinusoid", "direction_changes", "bursts", "stop_and_go", "roll", "throw")
STILL, WALK, SINE, TURNS, BURSTS, STOPGO, ROLL, THROW = range(len(FAMILIES))
AXIS_SCALE = np.array([1.0, 0.3])      # people and rolling balls move mostly sideways
BURST_FACTOR = 2.0                      # a burst: walking -> jogging, running -> sprinting
WALK_TAU_S = 1.0                        # velocity correlation time of the random walk
ROLL_TAU_S = 2.0                        # a rolling ball loses 63 % of its speed in this long
G_MPS2 = 9.81
THROW_APEX_M = (0.3, 1.5)               # height of a throw above where it starts
THROW_APEX_MAX_DEG = 25.0               # (keeps it inside the tilt range)
BOUNCE_E = (0.5, 0.75)                  # share of the vertical speed kept in a bounce
REST_APEX_DEG = 0.5                     # bounces lower than this: the ball comes to rest


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
        self.dist = np.ones(n)                  # target distance, m
        self.g = np.zeros(n)                    # throw: gravity, deg/s^2 at the target's distance
        self.vy = np.zeros(n)                   # vertical speed at the start of the step, deg/s
        self.height = np.zeros(n)               # deg above the level it was thrown from
        self.flying = np.zeros(n, bool)


    def reset(self, idx, family, speed, distance_m):
        k, rng = len(idx), self.rng
        self.family[idx] = family
        self.dist[idx] = distance_m
        self.g[idx] = np.degrees(G_MPS2 / distance_m)
        self.vy[idx], self.height[idx], self.flying[idx] = 0.0, 0.0, False
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
        self.vel[m] = BURST_FACTOR * s[m] * self.directions(m)
        self.burst_until[m] = t[m] + rng.uniform(0.2, 0.6, m.sum())
        self.next_event[m] = self.burst_until[m] + rng.uniform(1.0, 4.0, m.sum())
        walk((f == BURSTS) & (t >= self.burst_until), 0.3)

        m = (f == STOPGO) & event               # alternate moving and standing
        self.moving[m] = ~self.moving[m]
        go = m & self.moving
        self.vel[go] = s[go] * self.directions(go)
        self.vel[m & ~self.moving] = 0.0
        self.next_event[m] = t[m] + rng.uniform(0.5, 2.0, m.sum())

        m = f == ROLL                           # kicked, then slows down
        self.vel[m] *= np.exp(-dt[m, None] / ROLL_TAU_S)
        m &= event
        self.vel[m] = s[m] * self.directions(m) * rng.uniform(0.5, 1.5, (m.sum(), 1))
        self.next_event[m] = t[m] + rng.uniform(1.0, 4.0, m.sum())

        self.throw(f == THROW, t, dt, event)
        return self.vel


    def throw(self, m, t, dt, event):
        rng = self.rng
        go = m & ~self.flying & event           # thrown across, up to the apex height
        n = go.sum()
        apex = np.minimum(np.degrees(rng.uniform(*THROW_APEX_M, n) / self.dist[go]), THROW_APEX_MAX_DEG)
        self.vy[go] = np.sqrt(2 * self.g[go] * apex)
        self.vel[go, 0] = self.speed[go] * rng.uniform(0.5, 1.0, n) * rng.choice([-1, 1], n)
        self.height[go], self.flying[go] = 0.0, True

        fly = m & self.flying                   # exact parabola: the step's velocity is its mid-step value
        g, h = self.g[fly], dt[fly]
        self.vel[fly, 1] = self.vy[fly] - g * h / 2
        self.height[fly] += self.vel[fly, 1] * h
        self.vy[fly] -= g * h
        land = fly & (self.height < 0) & (self.vy < 0)
        self.height[land] = 0.0
        self.vy[land] *= -rng.uniform(*BOUNCE_E, land.sum())
        self.vel[land, 0] *= 0.8
        rest = land & (self.vy < np.sqrt(2 * self.g * REST_APEX_DEG))
        self.flying[rest], self.vy[rest] = False, 0.0
        self.vel[rest] = 0.0
        self.next_event[rest] = t[rest] + rng.uniform(0.3, 1.5, rest.sum())
        self.vel[m & ~self.flying] = 0.0



    def bounce(self, at_lo, at_hi):
        """Turn the velocity around on axes where the target hit the edge of its range"""
        out = (at_lo & (self.vel < 0)) | (at_hi & (self.vel > 0))
        self.vel[out] *= -1
