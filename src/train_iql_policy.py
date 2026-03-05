import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass
from pathlib import Path

from dqn.q_network import QNetwork
from dqn.v_network import VNetwork
from dqn.policiy_network import PolicyNet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# Config
# ============================================================

@dataclass
class PolicyConfig:

    train_npz: str
    iql_checkpoint: str
    output_policy: str

    batch_size: int
    policy_epochs: int

    lr_pi: float
    grad_clip_norm: float

    beta: float
    max_weight: float

    hidden1: int
    hidden2: int


# ============================================================
# Dataset
# ============================================================

def load_dataset(npz_path, batch_size):

    d = np.load(npz_path)

    obs = torch.tensor(d["observations"], dtype=torch.float32)
    actions = torch.tensor(d["actions"].reshape(-1), dtype=torch.long)

    dataset = torch.utils.data.TensorDataset(obs, actions)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )

    return loader


# ============================================================
# Train policy (AWBC)
# ============================================================

def train_policy(cfg: PolicyConfig):

    ckpt = torch.load(cfg.iql_checkpoint, map_location=DEVICE)

    state_dim = ckpt["state_dim"]
    num_actions = ckpt["num_actions"]

    Q1 = QNetwork(state_dim, num_actions, cfg.hidden1, cfg.hidden2).to(DEVICE)
    Q2 = QNetwork(state_dim, num_actions, cfg.hidden1, cfg.hidden2).to(DEVICE)
    V = VNetwork(state_dim, cfg.hidden1, cfg.hidden2).to(DEVICE)

    Q1.load_state_dict(ckpt["Q1"])
    Q2.load_state_dict(ckpt["Q2"])
    V.load_state_dict(ckpt["V"])

    Q1.eval()
    Q2.eval()
    V.eval()

    for p in Q1.parameters():
        p.requires_grad = False

    for p in Q2.parameters():
        p.requires_grad = False

    for p in V.parameters():
        p.requires_grad = False

    policy = PolicyNet(state_dim, num_actions, cfg.hidden1, cfg.hidden2).to(DEVICE)

    optimizer = optim.Adam(policy.parameters(), lr=cfg.lr_pi)

    loader = load_dataset(cfg.train_npz, cfg.batch_size)

    for epoch in range(1, cfg.policy_epochs + 1):

        loss_sum = 0
        adv_vals = []
        weight_vals = []
        entropy_vals = []

        pbar = tqdm(loader)

        for s, a in pbar:

            s = s.to(DEVICE)
            a = a.to(DEVICE)

            # ------------------------------------------------
            # Advantage computation
            # ------------------------------------------------

            with torch.no_grad():

                q1 = Q1(s)
                q2 = Q2(s)

                q = torch.minimum(q1, q2)

                q_sa = q.gather(1, a.unsqueeze(1)).squeeze(1)

                v = V(s)

                adv = q_sa - v

                weights = torch.exp(cfg.beta * adv)

                weights = torch.clamp(weights, max=cfg.max_weight)

                weights = weights / (weights.mean() + 1e-8)

            # ------------------------------------------------
            # Policy forward
            # ------------------------------------------------

            logits = policy(s)

            logp = torch.log_softmax(logits, dim=1)

            chosen_logp = logp.gather(1, a.unsqueeze(1)).squeeze(1)

            loss = -(weights * chosen_logp).mean()

            optimizer.zero_grad()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.grad_clip_norm)

            optimizer.step()

            loss_sum += loss.item()

            adv_vals.append(adv.cpu())
            weight_vals.append(weights.cpu())

            probs = torch.softmax(logits, dim=1)

            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1)

            entropy_vals.append(entropy.detach().cpu())

        adv_vals = torch.cat(adv_vals)
        weight_vals = torch.cat(weight_vals)
        entropy_vals = torch.cat(entropy_vals)

        print(
            f"[POLICY] epoch {epoch} | "
            f"loss {loss_sum/len(loader):.6f} | "
            f"adv_std {adv_vals.std():.3f} | "
            f"w_max {weight_vals.max():.3f} | "
            f"entropy {entropy_vals.mean():.3f}"
        )

    torch.save(
        {
            "policy": policy.state_dict(),
            "state_dim": state_dim,
            "num_actions": num_actions
        },
        cfg.output_policy
    )

    print("Policy saved to:", cfg.output_policy)


# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--config")

    args = parser.parse_args()

    with open(args.config) as f:
        cfg = PolicyConfig(**yaml.safe_load(f))

    train_policy(cfg)


if __name__ == "__main__":
    main()