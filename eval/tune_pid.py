#!/usr/bin/env python3
"""Tune the PID baseline in the simulator, then test it on the held-out scenarios.

Tuning maximizes the same reward the RL policy will be trained on (eval/evaluate.py scores), on
training seeds and difficulty 0.5. Many gain sets run at once: one group of envs per set.
  1. random search over kp, ki, kd, i_max_deg (integral limit) and hold_s (lost-target follow time)
  2. local search around the best few
  3. re-check the best with more envs and full-length episodes
Then the winner, a plain P controller and doing nothing are compared on the test scenarios
(other seeds, difficulty 0 and 1) and everything is saved.

    .venv/bin/python -m eval.tune_pid
"""

import argparse
import dataclasses
import datetime as dt
import json
import time
from pathlib import Path

import numpy as np

from eval.evaluate import compare, evaluate, report, report_compare, rollout, scenarios, scores, to_json
from eval.pid import PID, PIDGains

ROOT = Path(__file__).resolve().parents[1]


class NoAction:
    def __init__(self, n):
        self.n = n

    def reset(self, mask = None):
        pass

    def __call__(self, obs):
        return np.zeros((self.n, 2))


SEARCHED = ("kp", "ki", "kd", "i_max_deg", "hold_s")


def score_candidates(cands, envs_per, seconds, seed, scen):
    """Mean score per candidate over the training scenarios. cands: dict of arrays, keys SEARCHED"""
    n_c = len(cands["kp"])
    rep = {k: np.repeat(v, envs_per) for k, v in cands.items()}
    total = np.zeros(n_c)
    for k, cfg in enumerate(scen.values()):
        pid = PID(n_c * envs_per, PIDGains(**rep))
        s = scores(rollout(cfg, pid, n_c * envs_per, seconds, seed + k))
        total += s.reshape(n_c, envs_per).mean(axis = 1)
    return total / len(scen)


def log_uniform(rng, lo, hi, n):
    return np.exp(rng.uniform(np.log(lo), np.log(hi), n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--random", type = int, default = 192, help = "random gain sets")
    ap.add_argument("--local", type = int, default = 96, help = "local search gain sets")
    ap.add_argument("--seed", type = int, default = 0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    train = scenarios(difficulties = (0.5,))
    t0 = time.time()

    n = args.random
    default = PIDGains()
    cands = dict(kp = log_uniform(rng, 0.03, 1.0, n),
                 ki = np.where(rng.random(n) < 0.5, 0.0, log_uniform(rng, 0.05, 5.0, n)),
                 kd = np.where(rng.random(n) < 0.5, 0.0, log_uniform(rng, 0.002, 0.1, n)),
                 i_max_deg = log_uniform(rng, 0.3, 5.0, n),
                 hold_s = rng.uniform(0.0, 0.6, n))
    for k in SEARCHED[1:]:                                                          # the plain P controllers
        cands[k][:2] = 0.0 if k in ("ki", "kd") else getattr(default, k)
    cands["kp"][:2] = (0.15, 0.3)
    s1 = score_candidates(cands, 24, 10.0, 0, train)
    print(f"[1] random search, {n} gain sets: best score {s1.max():+.4f} ({time.time() - t0:.0f} s)", flush = True)

    top = np.argsort(s1)[::-1][:4]
    per = args.local // len(top)
    m = len(top) * per
    jitter = lambda v: np.repeat(v[top], per) * np.exp(rng.normal(0, 0.3, m))      # relative, keeps zeros at zero
    local = {k: jitter(cands[k]) for k in SEARCHED if k != "hold_s"}
    local["hold_s"] = np.clip(np.repeat(cands["hold_s"][top], per) + rng.normal(0, 0.08, m), 0.0, None)
    s2 = score_candidates(local, 24, 10.0, 0, train)
    print(f"[2] local search, {len(s2)} gain sets: best score {s2.max():+.4f} ({time.time() - t0:.0f} s)", flush = True)

    pool = {k: np.concatenate([cands[k], local[k]]) for k in cands}
    pool_s = np.concatenate([s1, s2])
    best8 = np.argsort(pool_s)[::-1][:8]
    finals = {k: v[best8] for k, v in pool.items()}
    s3 = score_candidates(finals, 128, 20.0, 100, train)
    w = int(np.argmax(s3))
    gains = PIDGains(**{k: float(finals[k][w]) for k in SEARCHED})
    print(f"[3] re-check of the best 8 with 128 envs x 20 s:")
    for i in np.argsort(s3)[::-1]:
        print(f"      kp {finals['kp'][i]:.3f}  ki {finals['ki'][i]:.3f}  kd {finals['kd'][i]:.4f}"
              f"  i_max {finals['i_max_deg'][i]:.2f}  hold {finals['hold_s'][i]:.2f}  score {s3[i]:+.4f}"
              + ("   <- chosen" if i == w else ""))
    print(f"    ({time.time() - t0:.0f} s)\n", flush = True)

    results = {"no action": evaluate(NoAction),
               "P 0.3": evaluate(lambda n: PID(n, PIDGains(kp = 0.3))),
               "PID tuned": evaluate(lambda n: PID(n, gains))}
    report(results)
    report_compare("PID tuned", "P 0.3", compare(results["PID tuned"], results["P 0.3"]))

    (ROOT / "eval/pid_gains.json").write_text(json.dumps(dataclasses.asdict(gains), indent = 2))
    out = ROOT / "eval/results" / f"{dt.date.today().isoformat()}_pid.json"
    out.parent.mkdir(parents = True, exist_ok = True)
    out.write_text(json.dumps(dict(gains = dataclasses.asdict(gains), test = {k: to_json(v) for k, v in results.items()},
                                   search = dict(random = {k: v.tolist() for k, v in cands.items()}, random_scores = s1.tolist(),
                                                 local = {k: v.tolist() for k, v in local.items()}, local_scores = s2.tolist(),
                                                 recheck = {k: v.tolist() for k, v in finals.items()}, recheck_scores = s3.tolist())),
                              indent = 1))
    print(f"\nwrote eval/pid_gains.json and {out.relative_to(ROOT)} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
