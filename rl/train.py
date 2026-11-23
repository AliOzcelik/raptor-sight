#!/usr/bin/env python3
"""Train the SAC tracking policy in the vectorized simulator (Phase 3; PROJECT.md 6).

Steps sim.core.BatchSim directly (many envs per step, much faster than the Gymnasium wrapper),
resets finished envs, and runs `--updates-per-step` gradient updates per simulator step. The critic
sees the privileged simulator state unless --symmetric. The discount is SimConfig.reward.gamma,
which also sets the failure penalty, so training and evaluation price a failure the same.

Every --eval-every updates the deterministic policy is scored on validation scenarios (the
evaluation scenarios at the training difficulty, but other seeds than eval/evaluate.py's test
seeds) against the tuned PID; the best one is saved as best.pt.

    .venv/bin/python -m rl.train --name nominal              # Phase 3: nominal parameters
    .venv/bin/python -m rl.evaluate rl/runs/nominal/best.pt  # test against the PID
    .venv/bin/tensorboard --logdir rl/runs
"""

import argparse
import dataclasses
import json
import subprocess
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

from eval.evaluate import rollout, scenarios, scores
from eval.pid import PID, PIDGains
from rl.policy import SACPolicy
from rl.sac import SAC, ReplayBuffer
from sim.config import SimConfig
from sim.core import BatchSim
from sim.obs import obs_names

ROOT = Path(__file__).resolve().parents[1]
VAL_SEED = 5000             # eval/evaluate.py tests from seed 1000, tune_pid.py trains on 0 and 100


def validate(make_policy, scen, n, seconds):
    """Mean score over the validation scenarios, and per scenario"""
    per = {name: float(scores(rollout(cfg, make_policy(n), n, seconds, VAL_SEED + k)).mean())
           for k, (name, cfg) in enumerate(scen.items())}
    return float(np.mean(list(per.values()))), per


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd = ROOT, capture_output = True,
                              text = True, check = True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main():
    ap = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default = time.strftime("%Y-%m-%d_%H%M"))
    ap.add_argument("--difficulty", type = float, default = 0.0, help = "0: nominal parameters (Phase 3), 1: full randomization")
    ap.add_argument("--updates", type = int, default = 500_000, help = "gradient updates in total")
    ap.add_argument("--envs", type = int, default = 64)
    ap.add_argument("--updates-per-step", type = int, default = 4, help = "per simulator step (= per --envs transitions)")
    ap.add_argument("--batch", type = int, default = 512)
    ap.add_argument("--buffer", type = int, default = 1_000_000)
    ap.add_argument("--warmup", type = int, default = 20_000, help = "transitions with random actions before learning")
    ap.add_argument("--hidden", type = int, nargs = "+", default = [256, 256])
    ap.add_argument("--lr", type = float, default = 3e-4)
    ap.add_argument("--tau", type = float, default = 0.005)
    ap.add_argument("--symmetric", action = "store_true", help = "critic sees only the policy observation (ablation)")
    ap.add_argument("--eval-every", type = int, default = 25_000, help = "updates")
    ap.add_argument("--eval-envs", type = int, default = 128)
    ap.add_argument("--eval-seconds", type = float, default = 10.0)
    ap.add_argument("--log-every", type = int, default = 5_000, help = "updates")
    ap.add_argument("--seed", type = int, default = 0)
    ap.add_argument("--threads", type = int, default = 4, help = "torch CPU threads (4 is fastest on an M2 Pro)")
    ap.add_argument("--device", default = "cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    rng = np.random.default_rng(args.seed)
    cfg = dataclasses.replace(SimConfig(), difficulty = args.difficulty)
    sim = BatchSim(args.envs, cfg, seed = args.seed)
    n, gamma = args.envs, cfg.reward.gamma
    names = obs_names(cfg.action_history)
    crit_in = (lambda o, p: o) if args.symmetric else (lambda o, p: np.concatenate([o, p], axis = 1))

    obs = sim.obs.copy()
    priv = sim.privileged(sim.true_error()[0])
    obs_dim, crit_dim = obs.shape[1], crit_in(obs, priv).shape[1]
    agent = SAC(obs_dim, crit_dim, hidden = tuple(args.hidden), gamma = gamma, tau = args.tau, lr = args.lr,
                device = args.device)
    buf = ReplayBuffer(args.buffer, obs_dim, crit_dim)

    run = ROOT / "rl/runs" / args.name
    run.mkdir(parents = True, exist_ok = True)
    meta = dict(obs_names = list(names), action_history = cfg.action_history, hidden = list(args.hidden),
                difficulty = args.difficulty, gamma = gamma, symmetric = args.symmetric, commit = git_commit(),
                args = vars(args))
    (run / "config.json").write_text(json.dumps(meta, indent = 2))
    try:
        from torch.utils.tensorboard import SummaryWriter
        tb = SummaryWriter(str(run))
    except ImportError:
        tb = None

    val_scen = scenarios(difficulties = (args.difficulty,))
    gains = PIDGains(**json.loads((ROOT / "eval/pid_gains.json").read_text()))
    pid_val, _ = validate(lambda k: PID(k, gains), val_scen, args.eval_envs, args.eval_seconds)
    print(f"run {run.relative_to(ROOT)}: {n} envs, difficulty {args.difficulty:g}, critic "
          f"{'symmetric' if args.symmetric else 'asymmetric'} ({crit_dim} inputs), actor {obs_dim} -> {args.hidden} -> 2")
    print(f"validation: {len(val_scen)} scenarios x {args.eval_envs} envs x {args.eval_seconds:g} s; "
          f"tuned PID scores {pid_val:+.4f}\n", flush = True)

    ep_ret, ep_len = np.zeros(n), np.zeros(n, int)
    done_ret, done_fail = deque(maxlen = 500), deque(maxlen = 500)
    win_r, stats = [], {}
    best, updates, steps, t0 = -np.inf, 0, 0, time.time()
    next_log, next_eval = args.log_every, args.eval_every

    def save(path, val):
        torch.save(dict(meta, actor = agent.actor.state_dict(), updates = updates, transitions = steps * n,
                        val_score = val, pid_val_score = pid_val), path)

    while updates < args.updates:
        a = rng.uniform(-1, 1, (n, 2)) if len(buf) < args.warmup else agent.act(obs)
        obs2, r, term, trunc, info = sim.step(a)
        obs2 = obs2.copy()
        buf.add(obs, crit_in(obs, priv), a, r, term, obs2, crit_in(obs2, info["privileged"]))
        steps += 1
        ep_ret += r
        ep_len += 1
        win_r.append(info["reward_step"].mean())
        done = term | trunc
        if done.any():
            done_ret.extend(ep_ret[done])
            done_fail.extend(term[done])
            ep_ret[done], ep_len[done] = 0.0, 0
            sim.reset(done)
            obs = sim.obs.copy()
            priv = sim.privileged(sim.true_error()[0])
        else:
            obs, priv = obs2, info["privileged"]

        if len(buf) < args.warmup:
            continue
        for _ in range(args.updates_per_step):
            stats = agent.update(buf.sample(args.batch, args.device))
            updates += 1

        if updates >= next_log:
            next_log += args.log_every
            el = time.time() - t0
            line = dict(reward_step = float(np.mean(win_r)),
                        ep_return = float(np.mean(done_ret)) if done_ret else float("nan"),
                        fail_pct = 100 * float(np.mean(done_fail)) if done_fail else float("nan"), **stats)
            win_r = []
            print(f"{updates:8,d} upd {steps * n / 1e6:6.2f} M trans {el / 60:5.1f} min | r/step {line['reward_step']:+.4f}"
                  f"  ep return {line['ep_return']:+8.1f}  fail {line['fail_pct']:5.1f} % | alpha {stats['alpha']:.4f}"
                  f"  H {stats['entropy']:+.2f}  Q {stats['q']:+7.2f}  loss_q {stats['loss_q']:.3f}", flush = True)
            if tb:
                for k, v in line.items():
                    tb.add_scalar(f"train/{k}", v, updates)

        if updates >= next_eval:
            next_eval += args.eval_every
            actor = agent.actor.eval()
            val, per = validate(lambda k: SACPolicy(k, actor), val_scen, args.eval_envs, args.eval_seconds)
            agent.actor.train()
            mark = ""
            if val > best:
                best, mark = val, "  <- best, saved"
                save(run / "best.pt", val)
            save(run / "last.pt", val)
            worst = min(per, key = per.get)
            print(f"  validation {val:+.4f} (PID {pid_val:+.4f}, {val - pid_val:+.4f}); "
                  f"worst {worst} {per[worst]:+.3f}{mark}", flush = True)
            if tb:
                tb.add_scalar("val/score", val, updates)
                tb.add_scalar("val/minus_pid", val - pid_val, updates)

    torch.save(dict(meta, agent = agent.state_dict(), updates = updates), run / "agent_last.pt")
    print(f"\ndone: {updates:,d} updates, {steps * n:,d} transitions in {(time.time() - t0) / 60:.1f} min;"
          f" best validation {best:+.4f} vs PID {pid_val:+.4f} -> {(run / 'best.pt').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
