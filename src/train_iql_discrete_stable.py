import argparse
import yaml
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------------------------------------------
# Utils
# ------------------------------------------------------------

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


@torch.no_grad()
def polyak_update(target, online, tau):
    for t, o in zip(target.parameters(), online.parameters()):
        t.data.mul_(1 - tau)
        t.data.add_(tau * o.data)


def expectile_loss(diff, tau):
    weight = torch.where(diff > 0, tau, 1 - tau)
    return (weight * diff.pow(2)).mean()


# ------------------------------------------------------------
# Config
# ------------------------------------------------------------

@dataclass
class Config:

    experiment_name: str
    output_root: str
    seed: int

    train_npz: str

    gamma: float
    expectile: float
    tau_polyak: float
    reward_scale: float

    critic_epochs: int
    batch_size: int
    lr_q: float
    lr_v: float

    hidden1: int
    hidden2: int

    train_policy: bool
    policy_epochs: int
    lr_pi: float
    beta: float
    max_weight: float

    log_every_epochs: int


# ------------------------------------------------------------
# Networks
# ------------------------------------------------------------

class QNetwork(nn.Module):

    def __init__(self, state_dim, num_actions, h1, h2):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, num_actions)
        )

    def forward(self, s):

        return self.net(s)


class VNetwork(nn.Module):

    def __init__(self, state_dim, h1, h2):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, 1)
        )

    def forward(self, s):
        return self.net(s).squeeze(-1)


class PolicyNet(nn.Module):

    def __init__(self, state_dim, num_actions, h1, h2):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, num_actions)
        )

    def forward(self, s):
        return self.net(s)


# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------

def load_dataset(path, batch_size, reward_scale):

    d = np.load(path)

    obs = torch.tensor(d["observations"], dtype=torch.float32)
    next_obs = torch.tensor(d["next_observations"], dtype=torch.float32)
    actions = torch.tensor(d["actions"], dtype=torch.long)

    rewards = torch.tensor(d["rewards"], dtype=torch.float32) * reward_scale
    dones = torch.tensor(d["terminals"], dtype=torch.float32)

    num_actions = int(d["num_items"])

    dataset = torch.utils.data.TensorDataset(
        obs, actions, rewards, next_obs, dones
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )

    info = dict(
        N=len(obs),
        state_dim=obs.shape[1],
        num_actions=num_actions,
        reward_mean=float(rewards.mean()),
        reward_std=float(rewards.std())
    )

    return loader, info


# ------------------------------------------------------------
# Training
# ------------------------------------------------------------

def train_iql(cfg):

    loader, info = load_dataset(
        cfg.train_npz,
        cfg.batch_size,
        cfg.reward_scale
    )

    state_dim = info["state_dim"]
    num_actions = info["num_actions"]

    print("Dataset:", info)

    Q1 = QNetwork(state_dim, num_actions, cfg.hidden1, cfg.hidden2).to(DEVICE)
    Q2 = QNetwork(state_dim, num_actions, cfg.hidden1, cfg.hidden2).to(DEVICE)

    V = VNetwork(state_dim, cfg.hidden1, cfg.hidden2).to(DEVICE)

    Q1_t = QNetwork(state_dim, num_actions, cfg.hidden1, cfg.hidden2).to(DEVICE)
    Q2_t = QNetwork(state_dim, num_actions, cfg.hidden1, cfg.hidden2).to(DEVICE)
    V_t = VNetwork(state_dim, cfg.hidden1, cfg.hidden2).to(DEVICE)

    Q1_t.load_state_dict(Q1.state_dict())
    Q2_t.load_state_dict(Q2.state_dict())
    V_t.load_state_dict(V.state_dict())

    q_opt = optim.Adam(list(Q1.parameters()) + list(Q2.parameters()), lr=cfg.lr_q)
    v_opt = optim.Adam(V.parameters(), lr=cfg.lr_v)

    mse = nn.MSELoss()

    for epoch in range(1, cfg.critic_epochs + 1):

        q_loss_sum = 0
        v_loss_sum = 0

        for s, a, r, s_next, done in tqdm(loader, leave=False):

            s = s.to(DEVICE)
            a = a.to(DEVICE)
            r = r.to(DEVICE)
            s_next = s_next.to(DEVICE)
            done = done.to(DEVICE)

            # --------------------------
            # V update
            # --------------------------

            with torch.no_grad():

                q1 = Q1(s)
                q2 = Q2(s)

                q = torch.minimum(q1, q2)

                q_sa = q.gather(1, a.unsqueeze(1)).squeeze(1)

            v = V(s)

            v_loss = expectile_loss(q_sa - v, cfg.expectile)

            v_opt.zero_grad()
            v_loss.backward()
            v_opt.step()

            # --------------------------
            # Q update
            # --------------------------

            with torch.no_grad():
                target = r + cfg.gamma * (1 - done) * V_t(s_next)

            q1 = Q1(s).gather(1, a.unsqueeze(1)).squeeze(1)
            q2 = Q2(s).gather(1, a.unsqueeze(1)).squeeze(1)

            q_loss = mse(q1, target) + mse(q2, target)

            q_opt.zero_grad()
            q_loss.backward()
            q_opt.step()

            polyak_update(Q1_t, Q1, cfg.tau_polyak)
            polyak_update(Q2_t, Q2, cfg.tau_polyak)
            polyak_update(V_t, V, cfg.tau_polyak)

            q_loss_sum += q_loss.item()
            v_loss_sum += v_loss.item()

        print(
            f"[IQL] epoch {epoch:03d} | "
            f"Qloss {q_loss_sum/len(loader):.6f} | "
            f"Vloss {v_loss_sum/len(loader):.6f}"
        )


# ------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = Config(**yaml.safe_load(f))

    set_seed(cfg.seed)

    train_iql(cfg)


if __name__ == "__main__":
    main()