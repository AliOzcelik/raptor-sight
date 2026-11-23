"""Trained SAC actor with the controller interface of eval/evaluate.py (same as eval.pid.PID):

    make = load("rl/runs/<run>/best.pt")
    policy = make(n)             # n envs
    policy.reset()
    action = policy(obs)         # (n, 10 + 2 * action_history) -> (n, 2) in [-1, 1], deterministic
"""

import numpy as np
import torch

from rl.sac import Actor
from sim.obs import obs_names


class SACPolicy:

    def __init__(self, n, actor):
        self.n, self.actor = n, actor

    def reset(self, mask = None):
        pass                        # stateless: the action history is part of the observation

    @torch.no_grad()
    def __call__(self, obs):
        return self.actor(torch.as_tensor(obs, dtype = torch.float32)).numpy().astype(np.float64)




def load_actor(path):
    ck = torch.load(path, map_location = "cpu", weights_only = True)
    names = tuple(ck["obs_names"])
    if names != obs_names(ck["action_history"]):
        raise ValueError(f"{path}: observation layout {names} doesn't match sim/obs.py")
    actor = Actor(len(names), 2, tuple(ck["hidden"]))
    actor.load_state_dict(ck["actor"])
    return actor.eval(), ck


def load(path):
    """make_policy(n) for eval.evaluate.evaluate()"""
    actor, _ = load_actor(path)
    return lambda n: SACPolicy(n, actor)
