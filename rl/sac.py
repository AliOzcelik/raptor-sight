"""Soft Actor-Critic (PROJECT.md 6) with an asymmetric critic (PROJECT.md 5.4).

The actor sees only the policy observation (sim/obs.py), exactly what the Pi will have. The critic
can also see the simulator's privileged state (BatchSim.privileged(): true error, target velocity,
true servo angles, latency, this episode's parameters); it is only needed for training.

Observations are already scaled to about [-1, 1] by sim/obs.py, so no normalization statistics are
learned: the Pi feeds the actor the same from_tracker() output.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0


def mlp(sizes, out_act = None):
    layers = []
    for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
        layers.append(nn.Linear(a, b))
        if i < len(sizes) - 2:
            layers.append(nn.ReLU())
    if out_act:
        layers.append(out_act)
    return nn.Sequential(*layers)




class Actor(nn.Module):
    """obs -> tanh-squashed Gaussian over actions in [-1, 1]. Small enough for the Pi CPU (2 x 256)."""

    def __init__(self, obs_dim, act_dim = 2, hidden = (256, 256)):
        super().__init__()
        self.body = mlp((obs_dim, *hidden), nn.ReLU())
        self.head = nn.Linear(hidden[-1], 2 * act_dim)

    def dist(self, obs):
        mean, log_std = self.head(self.body(obs)).chunk(2, dim = -1)
        return mean, log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)

    def sample(self, obs):
        """Reparameterized sample and its log-probability (with the tanh correction)"""
        mean, log_std = self.dist(obs)
        u = mean + log_std.exp() * torch.randn_like(mean)
        logp = (-0.5 * ((u - mean) / log_std.exp()) ** 2 - log_std - 0.5 * math.log(2 * math.pi)).sum(-1)
        logp = logp - (2 * (math.log(2) - u - F.softplus(-2 * u))).sum(-1)     # log |d tanh(u) / du|
        return torch.tanh(u), logp

    def forward(self, obs):
        """Deterministic action (the deployed policy): tanh of the mean"""
        return torch.tanh(self.dist(obs)[0])




class EnsembleLinear(nn.Module):
    """k independent linear layers applied in one batched matmul: x (k, B, i) -> (k, B, o)"""

    def __init__(self, k, i, o):
        super().__init__()
        bound = 1 / math.sqrt(i)                         # nn.Linear's default init
        self.w = nn.Parameter(torch.empty(k, i, o).uniform_(-bound, bound))
        self.b = nn.Parameter(torch.empty(k, 1, o).uniform_(-bound, bound))

    def forward(self, x):
        return torch.baddbmm(self.b, x, self.w)




class Critics(nn.Module):
    """k Q-networks (k = 2: clipped double Q) evaluated together: (critic input, action) -> (k, B)"""

    def __init__(self, in_dim, act_dim = 2, hidden = (256, 256), k = 2):
        super().__init__()
        sizes = (in_dim + act_dim, *hidden, 1)
        self.k = k
        self.layers = nn.ModuleList(EnsembleLinear(k, a, b) for a, b in zip(sizes[:-1], sizes[1:]))

    def forward(self, x, a):
        h = torch.cat([x, a], dim = -1).expand(self.k, -1, -1)
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < len(self.layers) - 1:
                h = F.relu(h)
        return h.squeeze(-1)




class ReplayBuffer:
    """Ring buffer of transitions as CPU tensors. crit / crit2: critic input (obs, or obs + privileged)."""

    def __init__(self, size, obs_dim, crit_dim, act_dim = 2):
        z = lambda *s: torch.zeros(size, *s)
        self.obs, self.obs2 = z(obs_dim), z(obs_dim)
        self.crit, self.crit2 = z(crit_dim), z(crit_dim)
        self.act, self.rew, self.term = z(act_dim), z(), z()
        self.size, self.ptr, self.full = size, 0, False

    def add(self, obs, crit, act, rew, term, obs2, crit2):
        n = len(obs)
        idx = (torch.arange(n) + self.ptr) % self.size
        for buf, v in ((self.obs, obs), (self.crit, crit), (self.act, act), (self.rew, rew), (self.term, term),
                       (self.obs2, obs2), (self.crit2, crit2)):
            buf[idx] = torch.as_tensor(v, dtype = torch.float32)
        self.full |= self.ptr + n >= self.size
        self.ptr = (self.ptr + n) % self.size

    def __len__(self):
        return self.size if self.full else self.ptr

    def sample(self, batch, device):
        i = torch.randint(0, len(self), (batch,))
        return [b[i].to(device) for b in (self.obs, self.crit, self.act, self.rew, self.term, self.obs2, self.crit2)]




class SAC:

    def __init__(self, obs_dim, crit_dim, act_dim = 2, hidden = (256, 256), gamma = 0.99, tau = 0.005,
                 lr = 3e-4, target_entropy = None, device = "cpu"):
        self.device, self.gamma, self.tau = device, gamma, tau
        self.actor = Actor(obs_dim, act_dim, hidden).to(device)
        self.critic = Critics(crit_dim, act_dim, hidden).to(device)
        self.critic_targ = Critics(crit_dim, act_dim, hidden).to(device)
        self.critic_targ.load_state_dict(self.critic.state_dict())
        self.critic_targ.requires_grad_(False)
        self.log_alpha = torch.zeros((), device = device, requires_grad = True)
        self.target_entropy = -float(act_dim) if target_entropy is None else target_entropy
        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr = lr)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr = lr)
        self.opt_alpha = torch.optim.Adam([self.log_alpha], lr = lr)


    @torch.no_grad()
    def act(self, obs, deterministic = False):
        o = torch.as_tensor(obs, dtype = torch.float32, device = self.device)
        a = self.actor(o) if deterministic else self.actor.sample(o)[0]
        return a.cpu().numpy()


    def update(self, batch):
        obs, crit, act, rew, term, obs2, crit2 = batch
        alpha = self.log_alpha.exp().detach()

        with torch.no_grad():
            a2, logp2 = self.actor.sample(obs2)
            q2 = self.critic_targ(crit2, a2).min(0).values - alpha * logp2
            y = rew + self.gamma * (1 - term) * q2
        q = self.critic(crit, act)
        loss_q = ((q - y) ** 2).mean(1).sum()
        self.opt_critic.zero_grad(set_to_none = True)
        loss_q.backward()
        self.opt_critic.step()

        self.critic.requires_grad_(False)
        a, logp = self.actor.sample(obs)
        loss_pi = (alpha * logp - self.critic(crit, a).min(0).values).mean()
        self.opt_actor.zero_grad(set_to_none = True)
        loss_pi.backward()
        self.opt_actor.step()
        self.critic.requires_grad_(True)

        loss_alpha = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
        self.opt_alpha.zero_grad(set_to_none = True)
        loss_alpha.backward()
        self.opt_alpha.step()

        with torch.no_grad():
            for p, pt in zip(self.critic.parameters(), self.critic_targ.parameters()):
                pt.lerp_(p, self.tau)
        return dict(loss_q = loss_q.item(), loss_pi = loss_pi.item(), alpha = alpha.item(),
                    q = q.mean().item(), entropy = -logp.mean().item())


    def state_dict(self):
        return dict(actor = self.actor.state_dict(), critic = self.critic.state_dict(),
                    critic_targ = self.critic_targ.state_dict(), log_alpha = self.log_alpha.detach().cpu(),
                    opt_actor = self.opt_actor.state_dict(), opt_critic = self.opt_critic.state_dict(),
                    opt_alpha = self.opt_alpha.state_dict())
