"""Fixed test scenarios and the metrics of PROJECT.md 9.1, for any controller with the policy interface:

    policy = make_policy(n)      # n envs
    policy.reset()               # new episodes
    action = policy(obs)         # (n, 10 + 2 * action_history) observations -> (n, 2) actions in [-1, 1]

Each scenario runs n envs for a fixed time from a fixed seed, so different controllers see exactly the
same targets, parameters and noise (the simulator draws the same random numbers whatever the actions).
Envs whose episode fails (target out of frame too long) stop counting; their remaining steps count as
lost in the score.

Every score and failure rate comes with a 95% confidence half-width (`*_ci95`). To compare two
controllers, use compare(): it pairs the same episodes, which gives tighter intervals than comparing
the two tables.
"""

import dataclasses

import numpy as np

from sim.config import Param, SimConfig
from sim.core import BatchSim
from sim.obs import BASE_NAMES
from sim.trajectories import FAMILIES

SETTLE_BAND = 0.05          # error below this (half frame widths, both axes) counts as centred: the reward's eps (16 px)
REACQ_BAND = 0.1            # 32 px
REACQ_MIN_LOST_S = 0.2      # only losses at least this long count for re-acquisition


def only(cfg, kind, family, **target_changes):
    """cfg with every episode of one target kind ("person", "ball") and one motion family"""
    tp = cfg.target
    kinds = tuple(dataclasses.replace(k, family_weights = tuple(float(f == family) for f in FAMILIES))
                  if k.name == kind else k for k in tp.kinds)
    w = tuple(float(k.name == kind) for k in tp.kinds)
    return dataclasses.replace(cfg, target = dataclasses.replace(tp, kinds = kinds, kind_weights = w, **target_changes))


def scenarios(base = SimConfig(), difficulties = (0.0, 1.0)):
    """name -> SimConfig: 'kind/family@difficulty' for every motion family each kind uses, plus
    'step': a still person starting 80 % of the way to the frame edge, in a random direction, nominal parameters."""
    out = {"step": dataclasses.replace(only(base, "person", "still", init_offset = Param(0.8, 0.8, 0.8)),
                                       difficulty = 0.0)}
    for d in difficulties:
        for kind in base.target.kinds:
            for f, w in zip(FAMILIES, kind.family_weights):
                if w > 0 and f != "still":
                    out[f"{kind.name}/{f}@{d:g}"] = dataclasses.replace(only(base, kind.name, f), difficulty = d)
        out[f"person/still@{d:g}"] = dataclasses.replace(only(base, "person", "still"), difficulty = d)
    return out


def rollout(cfg, policy, n, seconds, seed):
    """Per-step arrays (T, n) of one fixed scenario. A policy with a bind(sim) method gets the simulator
    first (only reference controllers that cheat, like Oracle)."""
    sim = BatchSim(n, cfg, seed = seed)
    if hasattr(policy, "bind"):
        policy.bind(sim)
    policy.reset()
    obs = sim.obs
    T = int(round(seconds * cfg.fps))
    alive = np.ones(n, bool)
    rec = {k: [] for k in ("e", "in_frame", "valid", "a", "reward", "alive")}
    for _ in range(T):
        a = policy(obs)
        obs, r, term, _, info = sim.step(a)
        for k, v in (("e", info["e_true"]), ("in_frame", info["in_frame"]), ("a", a),
                     ("valid", obs[:, BASE_NAMES.index("valid")] > 0.5), ("reward", info["reward_step"]), ("alive", alive.copy())):
            rec[k].append(v)
        alive &= ~term
    out = {k: np.array(v) for k, v in rec.items()}
    out.update(fps = cfg.fps, w_l = cfg.reward.w_l, half = np.array(cfg.frame_size) / 2)
    return out


class Oracle:
    """Reference with perfect information: sees the true target angle and velocity, with no latency, detector
    or tracker. Not achievable; it shows how much of a scenario the mount can follow at all."""

    def __init__(self, n, lead_s = 0.06, gain = 0.6):
        self.n, self.lead_s, self.gain = n, lead_s, gain

    def bind(self, sim):
        self.sim = sim

    def reset(self, mask = None):
        pass

    def __call__(self, obs):
        s = self.sim
        mount = np.column_stack([s.p["mount_pan_deg"], s.p["mount_tilt_deg"]])
        goal = s.tgt + s.traj.vel * self.lead_s - mount
        return np.clip(self.gain * (goal - s.cmd) / s.cfg.servo.max_step_deg, -1, 1)


def scores(r):
    """Reward per step for each env; steps after a failed episode count as lost (-w_l)"""
    return np.where(r["alive"], r["reward"], -r["w_l"]).mean(axis = 0)


def err_hw(r):
    """(T, n) error size in half frame widths on both axes, like the reward"""
    return np.linalg.norm(r["e"] * r["half"], axis = 2) / r["half"][0]


def settling_times(r):
    """Per env: time after which |e| stays below SETTLE_BAND to the end (NaN if it never settles)"""
    inside = (err_hw(r) < SETTLE_BAND) & r["in_frame"] & r["alive"]
    T = len(inside)
    out = np.full(inside.shape[1], np.nan)
    for i in range(inside.shape[1]):
        bad = np.flatnonzero(~inside[:, i])
        if len(bad) == 0:
            out[i] = 0.0
        elif bad[-1] < T - 1:
            out[i] = (bad[-1] + 1) / r["fps"]
    return out


def reacquisition_times(r):
    """Time from the tracker finding the target again (after >= REACQ_MIN_LOST_S without it) until |e| < REACQ_BAND"""
    fps, out = r["fps"], []
    centred = err_hw(r) < REACQ_BAND
    for i in range(r["valid"].shape[1]):
        v, ok = r["valid"][:, i], r["alive"][:, i]
        lost = 0
        for k in range(len(v)):
            if not ok[k]:
                break
            if not v[k]:
                lost += 1
                continue
            if lost >= REACQ_MIN_LOST_S * fps:
                done = np.flatnonzero(centred[k:, i] & ok[k:])
                if len(done):
                    out.append(done[0] / fps)
            lost = 0
    return np.array(out)


Z95 = 1.96


def ci95(x):
    """95% half-width of the mean of the per-env values x"""
    return Z95 * float(np.std(x, ddof = 1)) / np.sqrt(len(x))


def metrics(r, step_test = False):
    alive = r["alive"]
    e_px = np.linalg.norm(r["e"] * r["half"], axis = 2)
    ok = alive & r["in_frame"]
    da = np.linalg.norm(np.diff(r["a"], axis = 0), axis = 2)
    s, failed = scores(r), ~alive[-1]
    m = dict(score = float(s.mean()), score_ci95 = ci95(s),
             err_px_mean = float(e_px[ok].mean()), err_px_p95 = float(np.percentile(e_px[ok], 95)),
             out_of_frame_pct = 100 * float((~r["in_frame"][alive]).mean()),
             failed_pct = 100 * float(failed.mean()), failed_ci95 = 100 * ci95(failed.astype(float)),
             smooth_da = float(da[alive[1:]].mean()),
             env_scores = s)                    # per env, for compare(); dropped by to_json()
    re = reacquisition_times(r)
    m.update(reacq_s = float(np.median(re)) if len(re) else None, reacq_n = len(re))
    if step_test:
        st = settling_times(r)
        m.update(settle_s = float(np.nanmedian(st)) if np.isfinite(st).any() else None,
                 settled_pct = 100 * float(np.isfinite(st).mean()))
    return m


def evaluate(make_policy, scen = None, n = 512, seconds = 20.0, seed = 1000):
    """name -> metrics for every scenario, plus 'mean score' over them and its 'mean score ci95'.
    n = 512 keeps a scenario's score within about +-0.04 and its failure rate within +-4 points (95%)."""
    scen = scen or scenarios()
    out = {}
    for k, (name, cfg) in enumerate(scen.items()):
        out[name] = metrics(rollout(cfg, make_policy(n), n, seconds, seed + k), step_test = name == "step")
    sc = scenario_names(out)
    out["mean score"] = float(np.mean([out[s]["score"] for s in sc]))
    out["mean score ci95"] = float(np.sqrt(np.sum([out[s]["score_ci95"] ** 2 for s in sc]))) / len(sc)
    return out


def scenario_names(res):
    return [s for s, v in res.items() if isinstance(v, dict)]


def compare(a, b):
    """Paired difference a - b of two evaluate() results on the same scenarios and seeds:
    {scenario: (mean, ci95)} plus 'mean score': (mean, ci95). An interval that excludes 0 is a real difference."""
    out = {}
    for s in scenario_names(a):
        d = a[s]["env_scores"] - b[s]["env_scores"]
        out[s] = (float(d.mean()), ci95(d))
    sc = list(out)
    out["mean score"] = (float(np.mean([out[s][0] for s in sc])),
                         float(np.sqrt(np.sum([out[s][1] ** 2 for s in sc]))) / len(sc))
    return out


def to_json(res):
    """evaluate() result without the per-env arrays"""
    return {s: ({k: v for k, v in m.items() if k != "env_scores"} if isinstance(m, dict) else m) for s, m in res.items()}


def report(results):
    """Table: one row per scenario, one column block per controller. results = {controller: evaluate(...)}"""
    names = list(results)
    scen = scenario_names(next(iter(results.values())))
    head = " score  ±95%   err px  out %  fail % ±95%   |da|  "
    print(f"{'':22s}" + "".join(f"|{n:^{len(head)}s}" for n in names))
    print(f"{'scenario':22s}" + f"|{head}" * len(names))
    for s in scen:
        row = f"{s:22s}"
        for n in names:
            m = results[n][s]
            row += (f"| {m['score']:+.3f} {m['score_ci95']:.3f}   {m['err_px_mean']:5.1f}  {m['out_of_frame_pct']:5.1f}"
                    f"  {m['failed_pct']:5.1f} {m['failed_ci95']:4.1f}   {m['smooth_da']:.3f} ")
        print(row)
    print(f"{'mean score':22s}" + "".join(f"| {results[n]['mean score']:+.3f} {results[n]['mean score ci95']:.3f}"
                                         + " " * (len(head) - 13) for n in names))
    for n in names:
        st = results[n]["step"]
        re = [results[n][s]["reacq_s"] for s in scen if results[n][s]["reacq_s"] is not None]
        settle = "never" if st["settle_s"] is None else f"{st['settle_s']:.2f} s"
        print(f"  {n}: step test settles in {settle} (median, {st['settled_pct']:.0f} % settle),"
              f" re-centred after a loss in {np.median(re) if re else float('nan'):.2f} s (median over scenarios)")


def report_compare(a_name, b_name, diff):
    """Print compare(a, b): per scenario a - b with its 95% interval, and whether it is a real difference"""
    print(f"\n{a_name} - {b_name}, paired over the same episodes (score difference, 95% interval):")
    for s, (d, c) in diff.items():
        verdict = "better" if d - c > 0 else "worse" if d + c < 0 else "no clear difference"
        print(f"  {s:22s} {d:+.3f} ± {c:.3f}   {verdict}")
