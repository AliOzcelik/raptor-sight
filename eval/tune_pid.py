#!/usr/bin/env python3
"""Tune the PID baseline in the simulator, then test it on the held-out scenarios.

Tuning maximizes the same reward the RL policy will be trained on (eval/evaluate.py scores), on
training seeds and difficulty 0.5. Many gain sets run at once: one group of envs per set.
  1. random search over kp, ki, kd
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

from eval.evaluate import evaluate, report, rollout, scenarios, scores
from eval.pid import PID, PIDGains

ROOT = Path(__file__).resolve().parents[1]


class NoAction:
    def __init__(self, n):
        self.n = n

    def reset(self, mask = None):
        pass

    def __call__(self, obs):
        return np.zeros((self.n, 2))


def score_candidates(cands, envs_per, seconds, seed, scen):
    """Mean score per candidate over the training scenarios. cands: dict of arrays kp, ki, kd"""
    n_c = len(cands["kp"])
    rep = {k: np.repeat(v, envs_per) for k, v in cands.items()}
    total = np.zeros(n_c)
    for k, cfg in enumerate(scen.values()):
        pid = PID(n_c * envs_per, PIDGains(kp = rep["kp"], ki = rep["ki"], kd = rep["kd"]))
        s = scores(rollout(cfg, pid, n_c * envs_per, seconds, seed + k))
        total += s.reshape(n_c, envs_per).mean(axis = 1)
    return total / len(scen)


def log_uniform(rng, lo, hi, n):
    return np.exp(rng.uniform(np.log(lo), np.log(hi), n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--random", type = int, default = 96, help = "random gain sets")
    ap.add_argument("--local", type = int, default = 64, help = "local search gain sets")
    ap.add_argument("--seed", type = int, default = 0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    train = scenarios(difficulties = (0.5,))
    t0 = time.time()

    n = args.random
    cands = dict(kp = log_uniform(rng, 0.03, 1.0, n),
                 ki = np.where(rng.random(n) < 0.5, 0.0, log_uniform(rng, 0.05, 5.0, n)),
                 kd = np.where(rng.random(n) < 0.5, 0.0, log_uniform(rng, 0.002, 0.1, n)))
    cands["kp"][:2], cands["ki"][:2], cands["kd"][:2] = (0.15, 0.3), 0.0, 0.0      # the plain P controllers
    s1 = score_candidates(cands, 24, 10.0, 0, train)
    print(f"[1] random search, {n} gain sets: best score {s1.max():+.4f} ({time.time() - t0:.0f} s)", flush = True)

    top = np.argsort(s1)[::-1][:4]
    per = args.local // len(top)
    jitter = lambda v, m: np.repeat(v[top], per) * np.exp(rng.normal(0, 0.3, len(top) * per)) * m
    local = dict(kp = jitter(cands["kp"], 1), ki = jitter(cands["ki"], 1), kd = jitter(cands["kd"], 1))
    s2 = score_candidates(local, 24, 10.0, 0, train)
    print(f"[2] local search, {len(s2)} gain sets: best score {s2.max():+.4f} ({time.time() - t0:.0f} s)", flush = True)

    pool = {k: np.concatenate([cands[k], local[k]]) for k in cands}
    pool_s = np.concatenate([s1, s2])
    best8 = np.argsort(pool_s)[::-1][:8]
    finals = {k: v[best8] for k, v in pool.items()}
    s3 = score_candidates(finals, 128, 20.0, 100, train)
    w = int(np.argmax(s3))
    gains = PIDGains(kp = float(finals["kp"][w]), ki = float(finals["ki"][w]), kd = float(finals["kd"][w]))
    print(f"[3] re-check of the best 8 with 128 envs x 20 s:")
    for i in np.argsort(s3)[::-1]:
        print(f"      kp {finals['kp'][i]:.3f}  ki {finals['ki'][i]:.3f}  kd {finals['kd'][i]:.4f}  score {s3[i]:+.4f}"
              + ("   <- chosen" if i == w else ""))
    print(f"    ({time.time() - t0:.0f} s)\n", flush = True)

    results = {"no action": evaluate(NoAction),
               "P 0.3": evaluate(lambda n: PID(n, PIDGains(kp = 0.3))),
               "PID tuned": evaluate(lambda n: PID(n, gains))}
    report(results)

    (ROOT / "eval/pid_gains.json").write_text(json.dumps(dataclasses.asdict(gains), indent = 2))
    out = ROOT / "eval/results" / f"{dt.date.today().isoformat()}_pid.json"
    out.parent.mkdir(parents = True, exist_ok = True)
    out.write_text(json.dumps(dict(gains = dataclasses.asdict(gains), test = results,
                                   search = dict(random = {k: v.tolist() for k, v in cands.items()}, random_scores = s1.tolist(),
                                                 local = {k: v.tolist() for k, v in local.items()}, local_scores = s2.tolist(),
                                                 recheck = {k: v.tolist() for k, v in finals.items()}, recheck_scores = s3.tolist())),
                              indent = 1))
    print(f"\nwrote eval/pid_gains.json and {out.relative_to(ROOT)} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
