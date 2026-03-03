import os
import json
import time
import argparse
from dataclasses import dataclass, asdict
from pathlib import Path

import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# CONFIG
# ============================================================

@dataclass
class IQLConfig:
    experiment_name: str
    train_npz: str
    output_root: str

    seed: int

    gamma: float
    expectile: float

    lr_q: float
    lr_v: float

    batch_size: int
    epochs: int

    emb_dim: int
    hidden1: int
    hidden2: int

    tau_polyak: float
    grad_clip_norm: float
    weight_decay: float

    log_diagnostics: bool


def load_config(path: str):
    with open(path) as f:
        return IQLConfig(**yaml.safe_load(f))


def create_run_dir(cfg: IQLConfig):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    cfg_str = json.dumps(asdict(cfg), sort_keys=True)
    cfg_hash = str(abs(hash(cfg_str)))[:6]

    run_name = f"{cfg.experiment_name}_{cfg_hash}_{timestamp}"
    run_dir = Path(cfg.output_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=False)

    with open(run_dir / "config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    return run_dir


# ============================================================
# NETWORKS
# ============================================================

class QNetwork(nn.Module):
    def __init__(self, num_items, state_dim, emb_dim, hidden1, hidden2):
        super().__init__()
        self.embedding = nn.Embedding(num_items + 1, emb_dim)

        self.net = nn.Sequential(
            nn.Linear(state_dim + emb_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 1),
        )

    def forward(self, s, a):
        a_emb = self.embedding(a)
        x = torch.cat([s, a_emb], dim=1)
        return self.net(x).squeeze(-1)


class VNetwork(nn.Module):
    def __init__(self, state_dim, hidden1, hidden2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 1),
        )

    def forward(self, s):
        return self.net(s).squeeze(-1)


# ============================================================
# LOSSES
# ============================================================

def expectile_loss(diff, tau):
    weight = torch.where(diff > 0, tau, 1.0 - tau)
    return (weight * diff.pow(2)).mean()


@torch.no_grad()
def polyak_update(target, online, tau):
    for p_t, p in zip(target.parameters(), online.parameters()):
        p_t.data.mul_(1.0 - tau)
        p_t.data.add_(tau * p.data)


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_dir = create_run_dir(cfg)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    print("=" * 60)
    print("IQL TRAINING (with diagnostics)")
    print("=" * 60)
    print("Config:", cfg)
    print("Run dir:", run_dir)
    print("Device:", DEVICE)
    print("=" * 60)

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    d = np.load(cfg.train_npz)

    obs = torch.tensor(d["observations"], dtype=torch.float32)
    next_obs = torch.tensor(d["next_observations"], dtype=torch.float32)
    actions = torch.tensor(d["actions"].reshape(-1), dtype=torch.long)
    rewards = torch.tensor(d["rewards"], dtype=torch.float32)
    terminals = torch.tensor(d["terminals"], dtype=torch.float32)

    N, state_dim = obs.shape
    num_items = int(actions.max()) + 1

    print("Transitions:", N)
    print("State dim:", state_dim)
    print("Num items:", num_items)
    print("Reward mean:", float(rewards.mean()))
    print("Reward std :", float(rewards.std()))
    print("=" * 60)

    dataset = torch.utils.data.TensorDataset(
        obs, actions, rewards, next_obs, terminals
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True
    )

    # --------------------------------------------------------
    # Models
    # --------------------------------------------------------

    Q = QNetwork(num_items, state_dim, cfg.emb_dim, cfg.hidden1, cfg.hidden2).to(DEVICE)
    V = VNetwork(state_dim, cfg.hidden1, cfg.hidden2).to(DEVICE)
    V_target = VNetwork(state_dim, cfg.hidden1, cfg.hidden2).to(DEVICE)
    V_target.load_state_dict(V.state_dict())
    V_target.eval()

    q_opt = optim.Adam(Q.parameters(), lr=cfg.lr_q, weight_decay=cfg.weight_decay)
    v_opt = optim.Adam(V.parameters(), lr=cfg.lr_v, weight_decay=cfg.weight_decay)

    mse = nn.MSELoss()

    diagnostics = []

    # ============================================================
    # TRAINING LOOP
    # ============================================================

    for epoch in range(1, cfg.epochs + 1):

        Q.train()
        V.train()

        q_loss_acc = 0.0
        v_loss_acc = 0.0

        adv_values = []

        pbar = tqdm(loader, desc=f"Epoch {epoch}/{cfg.epochs}", leave=False)

        for s, a, r, s_next, done in pbar:

            s = s.to(DEVICE)
            a = a.to(DEVICE)
            r = r.to(DEVICE)
            s_next = s_next.to(DEVICE)
            done = done.to(DEVICE)

            # ----------------------------------------------------
            # V UPDATE
            # ----------------------------------------------------

            with torch.no_grad():
                q_sa = Q(s, a)

            v_s = V(s)
            diff = q_sa - v_s
            v_loss = expectile_loss(diff, cfg.expectile)

            v_opt.zero_grad(set_to_none=True)
            v_loss.backward()
            nn.utils.clip_grad_norm_(V.parameters(), cfg.grad_clip_norm)
            v_opt.step()

            # ----------------------------------------------------
            # Q UPDATE
            # ----------------------------------------------------

            with torch.no_grad():
                v_next = V_target(s_next)
                target = r + cfg.gamma * (1.0 - done) * v_next

            q_pred = Q(s, a)
            q_loss = mse(q_pred, target)

            q_opt.zero_grad(set_to_none=True)
            q_loss.backward()
            nn.utils.clip_grad_norm_(Q.parameters(), cfg.grad_clip_norm)
            q_opt.step()

            polyak_update(V_target, V, cfg.tau_polyak)

            q_loss_acc += q_loss.item()
            v_loss_acc += v_loss.item()

            adv_values.append(diff.detach().cpu())

        # --------------------------------------------------------
        # Diagnostics
        # --------------------------------------------------------

        adv_all = torch.cat(adv_values)

        adv_mean = adv_all.mean().item()
        adv_std = adv_all.std().item()
        adv_min = adv_all.min().item()
        adv_max = adv_all.max().item()

        print(
            f"Epoch {epoch:02d} | "
            f"Q Loss: {q_loss_acc/len(loader):.4f} | "
            f"V Loss: {v_loss_acc/len(loader):.4f} | "
            f"Adv std: {adv_std:.6f}"
        )

        diagnostics.append({
            "epoch": epoch,
            "q_loss": q_loss_acc/len(loader),
            "v_loss": v_loss_acc/len(loader),
            "adv_mean": adv_mean,
            "adv_std": adv_std,
            "adv_min": adv_min,
            "adv_max": adv_max,
        })

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    torch.save(
        {
            "Q": Q.state_dict(),
            "V": V.state_dict(),
            "V_target": V_target.state_dict(),
            "state_dim": state_dim,
            "num_items": num_items,
            "config": asdict(cfg),
        },
        run_dir / "iql_checkpoint.pt"
    )

    with open(run_dir / "diagnostics.json", "w") as f:
        json.dump(diagnostics, f, indent=2)

    print("=" * 60)
    print("Training complete.")
    print("Saved checkpoint to:", run_dir / "iql_checkpoint.pt")
    print("Saved diagnostics to:", run_dir / "diagnostics.json")
    print("=" * 60)


if __name__ == "__main__":
    main()