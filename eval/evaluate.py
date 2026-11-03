"""Fixed test scenarios and the metrics of PROJECT.md 9.1, for any controller with the policy interface:

    policy = make_policy(n)      # n envs
    policy.reset()               # new episodes
    action = policy(obs)         # (n, 12) observations -> (n, 2) actions in [-1, 1]

Each scenario runs n envs for a fixed time from a fixed seed, so different controllers see the same
targets and parameters. Envs whose episode fails (target out of frame too long) stop counting;
their remaining steps count as lost in the score.
"""

import dataclasses

import numpy as np

from sim.config import Param, SimConfig
from sim.core import BatchSim
from sim.obs import OBS_NAMES
from sim.trajectories import FAMILIES

SETTLE_BAND = 0.05          # |e| below this counts as centred (same as the reward's eps)
REACQ_BAND = 0.1
REACQ_MIN_LOST_S = 0.2      # only losses at least this long count for re-acquisition


def only_family(cfg, family, **target_changes):
    w = tuple(float(f == family) for f in FAMILIES)
    return dataclasses.replace(cfg, target = dataclasses.replace(cfg.target, family_weights = w, **target_changes))


def scenarios(base = SimConfig(), difficulties = (0.0, 1.0)):
    """name -> SimConfig. 'step': still target starting 80 % of half the FOV off-centre, nominal parameters."""
    out = {"step": dataclasses.replace(only_family(base, "still", init_offset = Param(0.8, 0.8, 0.8)),
                                       difficulty = 0.0)}
    for d in difficulties:
        for f in FAMILIES:
            out[f"{f}@{d:g}"] = dataclasses.replace(only_family(base, f), difficulty = d)
    return out


def rollout(cfg, policy, n, seconds, seed):
    """Per-step arrays (T, n) of one fixed scenario"""
    sim = BatchSim(n, cfg, seed = seed)
    policy.reset()
    obs = sim.obs
    T = int(round(seconds * cfg.fps))
    alive = np.ones(n, bool)
    rec = {k: [] for k in ("e", "in_frame", "valid", "a", "reward", "alive")}
    for _ in range(T):
        a = policy(obs)
        obs, r, term, _, info = sim.step(a)
        for k, v in (("e", info["e_true"]), ("in_frame", info["in_frame"]), ("a", a),
                     ("valid", obs[:, OBS_NAMES.index("valid")] > 0.5), ("reward", r), ("alive", alive.copy())):
            rec[k].append(v)
        alive &= ~term
    out = {k: np.array(v) for k, v in rec.items()}
    out.update(fps = cfg.fps, w_l = cfg.reward.w_l, half = np.array(cfg.frame_size) / 2)
    return out


def scores(r):
    """Reward per step for each env; steps after a failed episode count as lost (-w_l)"""
    return np.where(r["alive"], r["reward"], -r["w_l"]).mean(axis = 0)


def settling_times(r):
    """Per env: time after which |e| stays below SETTLE_BAND to the end (NaN if it never settles)"""
    inside = (np.linalg.norm(r["e"], axis = 2) < SETTLE_BAND) & r["in_frame"] & r["alive"]
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
    centred = np.linalg.norm(r["e"], axis = 2) < REACQ_BAND
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


def metrics(r, step_test = False):
    alive = r["alive"]
    e_px = np.linalg.norm(r["e"] * r["half"], axis = 2)
    ok = alive & r["in_frame"]
    da = np.linalg.norm(np.diff(r["a"], axis = 0), axis = 2)
    m = dict(score = float(scores(r).mean()),
             err_px_mean = float(e_px[ok].mean()), err_px_p95 = float(np.percentile(e_px[ok], 95)),
             out_of_frame_pct = 100 * float((~r["in_frame"][alive]).mean()),
             failed_pct = 100 * float((~alive[-1]).mean()),
             smooth_da = float(da[alive[1:]].mean()))
    re = reacquisition_times(r)
    m.update(reacq_s = float(np.median(re)) if len(re) else None, reacq_n = len(re))
    if step_test:
        st = settling_times(r)
        m.update(settle_s = float(np.nanmedian(st)) if np.isfinite(st).any() else None,
                 settled_pct = 100 * float(np.isfinite(st).mean()))
    return m


def evaluate(make_policy, scen = None, n = 64, seconds = 20.0, seed = 1000):
    """name -> metrics for every scenario, plus 'mean score' over them"""
    scen = scen or scenarios()
    out = {}
    for k, (name, cfg) in enumerate(scen.items()):
        out[name] = metrics(rollout(cfg, make_policy(n), n, seconds, seed + k), step_test = name == "step")
    out["mean score"] = float(np.mean([v["score"] for v in out.values()]))
    return out


def report(results):
    """Table: one row per scenario, one column block per controller. results = {controller: evaluate(...)}"""
    names = list(results)
    scen = [s for s in next(iter(results.values())) if s != "mean score"]
    cols = (("score", "{:+.3f}"), ("err_px_mean", "{:5.1f}"), ("out_of_frame_pct", "{:5.1f}"),
            ("failed_pct", "{:5.1f}"), ("smooth_da", "{:.3f}"))
    print(f"{'':22s}" + "".join(f"| {n:^40s}" for n in names))
    print(f"{'scenario':22s}" + "| score  err px  out %  fail %  |da|  " * len(names))
    for s in scen:
        row = f"{s:22s}"
        for n in names:
            m = results[n][s]
            row += "| " + "  ".join(f.format(m[c]) for c, f in cols) + "   "
        print(row)
    print(f"{'mean score':22s}" + "".join(f"| {results[n]['mean score']:+.3f}{'':34s}" for n in names))
    for n in names:
        st = results[n]["step"]
        re = [results[n][s]["reacq_s"] for s in scen if results[n][s]["reacq_s"] is not None]
        print(f"  {n}: step test settles in {st['settle_s']} s (median, {st['settled_pct']:.0f} % settle),"
              f" re-centred after a loss in {np.median(re) if re else float('nan'):.2f} s (median over scenarios)")
