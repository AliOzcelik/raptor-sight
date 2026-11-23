#!/usr/bin/env python3
"""Test a trained SAC policy against the tuned PID on the held-out scenarios of eval/evaluate.py
(test seeds, difficulty 0 and 1, 512 episodes each) and save the results.

Phase 3 is done when SAC beats the PID on nominal parameters: the paired score difference over the
nominal scenarios (difficulty 0 and the step test) is positive with its 95% interval above 0.

    .venv/bin/python -m rl.evaluate rl/runs/nominal/best.pt
"""

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np
import torch

from eval.evaluate import compare, evaluate, report, report_compare, to_json
from eval.pid import PID, PIDGains
from rl.policy import load, load_actor

ROOT = Path(__file__).resolve().parents[1]


def subset_mean(diff, names):
    """Mean of compare()'s per-scenario differences over names, with its 95% interval"""
    return (float(np.mean([diff[s][0] for s in names])),
            float(np.sqrt(np.sum([diff[s][1] ** 2 for s in names]))) / len(names))


def main():
    ap = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("--n", type = int, default = 512, help = "episodes per scenario")
    ap.add_argument("--out", help = "results JSON (default: eval/results/<date>_sac_<run>.json)")
    args = ap.parse_args()
    torch.set_num_threads(4)

    ck_path = Path(args.checkpoint).resolve()
    _, ck = load_actor(ck_path)
    gains = PIDGains(**json.loads((ROOT / "eval/pid_gains.json").read_text()))
    results = {"PID tuned": evaluate(lambda n: PID(n, gains), n = args.n),
               "SAC": evaluate(load(ck_path), n = args.n)}
    report(results)
    diff = compare(results["SAC"], results["PID tuned"])
    report_compare("SAC", "PID tuned", diff)

    nominal = [s for s in diff if s == "step" or s.endswith("@0")]
    full = [s for s in diff if s.endswith("@1")]
    nom, rand = subset_mean(diff, nominal), subset_mean(diff, full)
    passed = nom[0] - nom[1] > 0
    print(f"\nSAC - PID, nominal parameters ({len(nominal)} scenarios): {nom[0]:+.4f} ± {nom[1]:.4f}")
    print(f"SAC - PID, full randomization ({len(full)} scenarios):  {rand[0]:+.4f} ± {rand[1]:.4f}")
    print(f"Phase 3 (SAC beats PID on nominal parameters): {'PASS' if passed else 'not yet'}")

    run = ck_path.parent.name
    out = Path(args.out) if args.out else ROOT / "eval/results" / f"{dt.date.today().isoformat()}_sac_{run}.json"
    out.parent.mkdir(parents = True, exist_ok = True)
    meta = {k: ck[k] for k in ("difficulty", "symmetric", "hidden", "updates", "transitions", "commit", "args") if k in ck}
    out.write_text(json.dumps(dict(checkpoint = str(ck_path.relative_to(ROOT)) if ck_path.is_relative_to(ROOT) else str(ck_path),
                                   training = meta, pid_gains = gains.__dict__,
                                   phase3 = dict(nominal_diff = nom, randomized_diff = rand, passed = passed),
                                   compare = diff, test = {k: to_json(v) for k, v in results.items()}), indent = 1))
    print(f"wrote {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}")


if __name__ == "__main__":
    main()
