import argparse
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import yaml
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# Utils
# ============================================================

def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def polyak_update(target: nn.Module, online: nn.Module, tau: float):
    for pt, p in zip(target.parameters(), online.parameters()):
        pt.data.mul_(1.0 - tau)
        pt.data.add_(tau * p.data)


def expectile_loss(diff: torch.Tensor, tau: float) -> torch.Tensor:
    # diff = Q(s,a) - V(s)
    w = torch.where(diff > 0, tau, 1.0 - tau)
    return (w * diff.pow(2)).mean()


def make_run_dir(output_root: str, experiment_name: str, cfg: dict) -> Path:
    ts = time.strftime("%Y%m%d_%H%M%S")
    cfg_hash = str(abs(hash(json.dumps(cfg, sort_keys=True))))[:6]
    run_dir = Path(output_root) / f"{experiment_name}_{cfg_hash}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    return run_dir


# ============================================================
# Config
# ============================================================

@dataclass
class Config:
    # run
    experiment_name: str
    output_root: str
    seed: int

    # data
    train_npz: str
    reward_scale: float

    # algo
    gamma: float
    expectile: float

    # training
    epochs: int
    batch_size: int
    lr_q: float
    lr_v: float
    weight_decay: float
    grad_clip_norm: float

    # target nets
    tau_polyak: float
    target_update_period: int  # update targets every K gradient steps

    # model
    hidden1: int
    hidden2: int

    # stability knobs
    huber_delta: float
    clamp_v_min: Optional[float]
    clamp_v_max: Optional[float]
    clamp_adv_min: Optional[float]
    clamp_adv_max: Optional[float]

    # logging
    log_every_epochs: int


# ============================================================
# Networks: vector-Q (DQN style)
# ============================================================

class QNetwork(nn.Module):
    """Outputs Q(s) for ALL discrete actions: shape (B, num_actions)."""
    def __init__(self, state_dim: int, num_actions: int, h1: int, h2: int):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, num_actions)

        # Stable init:
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity="relu")
        nn.init.zeros_(self.fc1.bias)
        nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity="relu")
        nn.init.zeros_(self.fc2.bias)

        # CRITICAL: start Q near 0
        nn.init.zeros_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(s))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class VNetwork(nn.Module):
    """Outputs scalar V(s): shape (B,)."""
    def __init__(self, state_dim: int, h1: int, h2: int):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, 1)

        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity="relu")
        nn.init.zeros_(self.fc1.bias)
        nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity="relu")
        nn.init.zeros_(self.fc2.bias)

        # CRITICAL: start V near 0
        nn.init.zeros_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(s))
        x = F.relu(self.fc2(x))
        return self.fc3(x).squeeze(-1)


# ============================================================
# Data
# ============================================================

def load_npz(npz_path: str, batch_size: int, reward_scale: float):
    d = np.load(npz_path)

    obs = torch.tensor(d["observations"], dtype=torch.float32)
    next_obs = torch.tensor(d["next_observations"], dtype=torch.float32)

    actions = d["actions"]
    if actions.ndim > 1:
        actions = actions.reshape(-1)
    actions = torch.tensor(actions, dtype=torch.long)

    rewards = torch.tensor(d["rewards"], dtype=torch.float32).reshape(-1) * float(reward_scale)
    terminals = torch.tensor(d["terminals"], dtype=torch.float32).reshape(-1)

    if "num_items" in d:
        num_actions = int(d["num_items"])
    else:
        num_actions = int(actions.max().item()) + 1

    # quick sanity
    if actions.min().item() < 0 or actions.max().item() >= num_actions:
        raise ValueError(f"actions out of range: min={actions.min().item()} max={actions.max().item()} num_actions={num_actions}")

    dataset = torch.utils.data.TensorDataset(obs, actions, rewards, next_obs, terminals)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    info = {
        "N": int(obs.shape[0]),
        "state_dim": int(obs.shape[1]),
        "num_actions": int(num_actions),
        "reward_mean": float(rewards.mean().item()),
        "reward_std": float(rewards.std().item()),
    }
    return loader, info


# ============================================================
# Train
# ============================================================

def train(cfg: Config):
    run_dir = make_run_dir(cfg.output_root, cfg.experiment_name, asdict(cfg))
    print("Run dir:", run_dir)

    loader, info = load_npz(cfg.train_npz, cfg.batch_size, cfg.reward_scale)
    print("Dataset:", info)

    state_dim = info["state_dim"]
    num_actions = info["num_actions"]

    Q1 = QNetwork(state_dim, num_actions, cfg.hidden1, cfg.hidden2).to(DEVICE)
    Q2 = QNetwork(state_dim, num_actions, cfg.hidden1, cfg.hidden2).to(DEVICE)
    V = VNetwork(state_dim, cfg.hidden1, cfg.hidden2).to(DEVICE)

    # targets
    Q1_t = QNetwork(state_dim, num_actions, cfg.hidden1, cfg.hidden2).to(DEVICE)
    Q2_t = QNetwork(state_dim, num_actions, cfg.hidden1, cfg.hidden2).to(DEVICE)
    V_t = VNetwork(state_dim, cfg.hidden1, cfg.hidden2).to(DEVICE)
    Q1_t.load_state_dict(Q1.state_dict())
    Q2_t.load_state_dict(Q2.state_dict())
    V_t.load_state_dict(V.state_dict())
    Q1_t.eval(); Q2_t.eval(); V_t.eval()

    q_opt = optim.Adam(
        list(Q1.parameters()) + list(Q2.parameters()),
        lr=cfg.lr_q,
        weight_decay=cfg.weight_decay,
        eps=1e-8,
    )
    v_opt = optim.Adam(V.parameters(), lr=cfg.lr_v, weight_decay=cfg.weight_decay, eps=1e-8)

    global_step = 0
    diag = {"epochs": []}

    for epoch in range(1, cfg.epochs + 1):
        Q1.train(); Q2.train(); V.train()

        q_loss_sum = 0.0
        v_loss_sum = 0.0
        adv_accum = []
        td_accum = []

        pbar = tqdm(loader, desc=f"[IQL-stable] epoch {epoch}/{cfg.epochs}", leave=False)
        for s, a, r, s_next, done in pbar:
            global_step += 1

            s = s.to(DEVICE)
            a = a.to(DEVICE)
            r = r.to(DEVICE)
            s_next = s_next.to(DEVICE)
            done = done.to(DEVICE)

            # ------------------------------------------------------------
            # (1) V update: expectile regression to min(Q1,Q2) at dataset action
            # IMPORTANT: stop-grad through Q (IQL requirement).
            # Use ONLINE Q (not target) for V fit (standard & stable).
            # ------------------------------------------------------------
            with torch.no_grad():
                q1_all = Q1(s)
                q2_all = Q2(s)
                q_all = torch.minimum(q1_all, q2_all)
                q_sa = q_all.gather(1, a.unsqueeze(1)).squeeze(1)

            v_s = V(s)
            diff = q_sa - v_s

            if cfg.clamp_adv_min is not None or cfg.clamp_adv_max is not None:
                mn = cfg.clamp_adv_min if cfg.clamp_adv_min is not None else float("-inf")
                mx = cfg.clamp_adv_max if cfg.clamp_adv_max is not None else float("inf")
                diff = diff.clamp(mn, mx)

            v_loss = expectile_loss(diff, cfg.expectile)

            v_opt.zero_grad(set_to_none=True)
            v_loss.backward()
            nn.utils.clip_grad_norm_(V.parameters(), cfg.grad_clip_norm)
            v_opt.step()

            # optional clamp V output range (extra stability for recsys)
            if cfg.clamp_v_min is not None or cfg.clamp_v_max is not None:
                with torch.no_grad():
                    for p in V.parameters():
                        pass  # no-op (kept for clarity)
                # Note: we clamp the VALUE predictions implicitly by clamping advantage above and using stable init.

            # ------------------------------------------------------------
            # (2) Q update: target = r + gamma * V_target(s')
            # IMPORTANT: no max over actions at s' (IQL core).
            # Use Huber for robustness.
            # ------------------------------------------------------------
            with torch.no_grad():
                v_next = V_t(s_next)
                if cfg.clamp_v_min is not None or cfg.clamp_v_max is not None:
                    mnv = cfg.clamp_v_min if cfg.clamp_v_min is not None else float("-inf")
                    mxv = cfg.clamp_v_max if cfg.clamp_v_max is not None else float("inf")
                    v_next = v_next.clamp(mnv, mxv)
                target = r + cfg.gamma * (1.0 - done) * v_next

            q1_sa_pred = Q1(s).gather(1, a.unsqueeze(1)).squeeze(1)
            q2_sa_pred = Q2(s).gather(1, a.unsqueeze(1)).squeeze(1)

            q_loss = F.huber_loss(q1_sa_pred, target, delta=cfg.huber_delta) + \
                     F.huber_loss(q2_sa_pred, target, delta=cfg.huber_delta)

            q_opt.zero_grad(set_to_none=True)
            q_loss.backward()
            nn.utils.clip_grad_norm_(list(Q1.parameters()) + list(Q2.parameters()), cfg.grad_clip_norm)
            q_opt.step()

            # ------------------------------------------------------------
            # (3) Target update (SLOW + PERIODIC)
            # This is where your previous run can explode: target drift too fast.
            # ------------------------------------------------------------
            if (global_step % int(cfg.target_update_period)) == 0:
                polyak_update(Q1_t, Q1, cfg.tau_polyak)
                polyak_update(Q2_t, Q2, cfg.tau_polyak)
                polyak_update(V_t, V, cfg.tau_polyak)

            q_loss_sum += float(q_loss.item())
            v_loss_sum += float(v_loss.item())

            with torch.no_grad():
                adv = (q_sa - v_s).detach().cpu()
                adv_accum.append(adv)
                td = (target - torch.minimum(q1_sa_pred, q2_sa_pred)).detach().cpu()
                td_accum.append(td)

        adv_all = torch.cat(adv_accum)
        td_all = torch.cat(td_accum)

        row = {
            "epoch": epoch,
            "q_loss": q_loss_sum / len(loader),
            "v_loss": v_loss_sum / len(loader),
            "adv_mean": float(adv_all.mean().item()),
            "adv_std": float(adv_all.std().item()),
            "adv_min": float(adv_all.min().item()),
            "adv_max": float(adv_all.max().item()),
            "td_mean": float(td_all.mean().item()),
            "td_std": float(td_all.std().item()),
        }
        diag["epochs"].append(row)

        if epoch % max(1, cfg.log_every_epochs) == 0:
            print(
                f"[IQL-stable] epoch {epoch:03d} | "
                f"Qloss {row['q_loss']:.6f} | Vloss {row['v_loss']:.6f} | "
                f"Adv std {row['adv_std']:.6f} | TD std {row['td_std']:.6f}"
            )

        # Hard stop if NaN (better fail-fast)
        if not np.isfinite(row["q_loss"]) or not np.isfinite(row["v_loss"]):
            raise RuntimeError("Divergence detected (NaN/Inf). Check reward_scale/gamma/lr and dataset sanity.")

    # save
    torch.save(
        {
            "Q1": Q1.state_dict(),
            "Q2": Q2.state_dict(),
            "V": V.state_dict(),
            "Q1_target": Q1_t.state_dict(),
            "Q2_target": Q2_t.state_dict(),
            "V_target": V_t.state_dict(),
            "state_dim": state_dim,
            "num_actions": num_actions,
            "config": asdict(cfg),
        },
        run_dir / "iql_stable.pt",
    )
    (run_dir / "diagnostics.json").write_text(json.dumps(diag, indent=2))
    print("Saved:", run_dir / "iql_stable.pt")
    print("Saved:", run_dir / "diagnostics.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg_dict = yaml.safe_load(f)
    cfg = Config(**cfg_dict)

    set_seed(cfg.seed)
    print("=" * 80)
    print("IQL DISCRETE STABLE (large action space)")
    print("Device:", DEVICE)
    print("Config:", cfg)
    print("=" * 80)

    train(cfg)


if __name__ == "__main__":
    main()