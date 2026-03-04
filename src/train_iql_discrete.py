import os
import json
import time
import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict, Any

import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


# ============================================================
# Utils
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def polyak_update(target: nn.Module, online: nn.Module, tau: float):
    """EMA update: target = (1-tau)*target + tau*online"""
    for p_t, p in zip(target.parameters(), online.parameters()):
        p_t.data.mul_(1.0 - tau)
        p_t.data.add_(tau * p.data)


def expectile_loss(diff: torch.Tensor, tau: float) -> torch.Tensor:
    """
    Expectile regression loss:
      L = E[ w(diff) * diff^2 ], w = tau if diff>0 else (1-tau)
    where diff = (Q - V).
    """
    weight = torch.where(diff > 0, tau, 1.0 - tau)
    return (weight * diff.pow(2)).mean()


# ============================================================
# Config
# ============================================================

@dataclass
class IQLDiscreteConfig:
    # run
    experiment_name: str
    output_root: str
    seed: int

    # data
    train_npz: str

    # algo (IQL)
    gamma: float
    expectile: float
    tau_polyak: float

    # critic/value training
    critic_epochs: int
    batch_size: int
    lr_q: float
    lr_v: float
    weight_decay: float
    grad_clip_norm: float

    # model
    emb_dim: int
    hidden1: int
    hidden2: int

    # policy extraction (AWBC / "AWR-like" as used in IQL paper)
    train_policy: bool
    policy_epochs: int
    lr_pi: float
    beta: float              # IMPORTANT: used as exp(beta * advantage)
    max_weight: float
    policy_grad_clip: float

    # optional stabilizers (OFF by default for faithfulness)
    clamp_adv_min: Optional[float]
    clamp_adv_max: Optional[float]

    # logging
    log_every_epochs: int


def load_config(path: str) -> IQLDiscreteConfig:
    with open(path, "r") as f:
        d = yaml.safe_load(f)
    return IQLDiscreteConfig(**d)


def create_run_dir(cfg: IQLDiscreteConfig) -> Path:
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
# Networks (Discrete)
# ============================================================

class QNetwork(nn.Module):
    """
    Discrete-action Q(s,a) implemented with:
      - action embedding
      - MLP over [state, action_emb]
    This is valid for discrete actions and matches your existing approach.
    """
    def __init__(self, num_actions: int, state_dim: int, emb_dim: int, hidden1: int, hidden2: int):
        super().__init__()
        self.embedding = nn.Embedding(num_actions, emb_dim)
        self.net = nn.Sequential(
            nn.Linear(state_dim + emb_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 1),
        )

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        a_emb = self.embedding(a)
        x = torch.cat([s, a_emb], dim=1)
        return self.net(x).squeeze(-1)


class VNetwork(nn.Module):
    """State-value V(s)."""
    def __init__(self, state_dim: int, hidden1: int, hidden2: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 1),
        )

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.net(s).squeeze(-1)


class PolicyNet(nn.Module):
    """
    Discrete policy π(a|s) parameterized by logits over actions.
    Training objective (IQL policy extraction / AWBC):
      maximize E[ exp(beta * A(s,a)) * log π(a|s) ] over dataset actions.
    """
    def __init__(self, state_dim: int, num_actions: int, hidden1: int, hidden2: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, num_actions),
        )

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.net(s)  # logits


# ============================================================
# Training
# ============================================================

def make_dataloader_from_npz(npz_path: str, batch_size: int):
    d = np.load(npz_path)

    obs = torch.tensor(d["observations"], dtype=torch.float32)
    next_obs = torch.tensor(d["next_observations"], dtype=torch.float32)
    actions = torch.tensor(d["actions"].reshape(-1), dtype=torch.long)
    rewards = torch.tensor(d["rewards"], dtype=torch.float32)
    terminals = torch.tensor(d["terminals"], dtype=torch.float32)

    # prefer explicit num_items/num_actions from file if present (matches your current pipeline)
    if "num_items" in d:
        num_actions = int(d["num_items"])
    else:
        num_actions = int(actions.max().item()) + 1

    N, state_dim = obs.shape

    dataset = torch.utils.data.TensorDataset(obs, actions, rewards, next_obs, terminals)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    info = {
        "N": N,
        "state_dim": state_dim,
        "num_actions": num_actions,
        "reward_mean": float(rewards.mean().item()),
        "reward_std": float(rewards.std().item()),
    }
    return loader, info


def train_critics_and_value(
    cfg: IQLDiscreteConfig,
    loader,
    state_dim: int,
    num_actions: int,
    run_dir: Path,
):
    # Online nets
    Q1 = QNetwork(num_actions, state_dim, cfg.emb_dim, cfg.hidden1, cfg.hidden2).to(DEVICE)
    Q2 = QNetwork(num_actions, state_dim, cfg.emb_dim, cfg.hidden1, cfg.hidden2).to(DEVICE)
    V = VNetwork(state_dim, cfg.hidden1, cfg.hidden2).to(DEVICE)

    # Target nets (EMA)
    Q1_t = QNetwork(num_actions, state_dim, cfg.emb_dim, cfg.hidden1, cfg.hidden2).to(DEVICE)
    Q2_t = QNetwork(num_actions, state_dim, cfg.emb_dim, cfg.hidden1, cfg.hidden2).to(DEVICE)
    V_t = VNetwork(state_dim, cfg.hidden1, cfg.hidden2).to(DEVICE)
    Q1_t.load_state_dict(Q1.state_dict())
    Q2_t.load_state_dict(Q2.state_dict())
    V_t.load_state_dict(V.state_dict())
    Q1_t.eval(); Q2_t.eval(); V_t.eval()

    q_params = list(Q1.parameters()) + list(Q2.parameters())
    q_opt = optim.Adam(q_params, lr=cfg.lr_q, weight_decay=cfg.weight_decay)
    v_opt = optim.Adam(V.parameters(), lr=cfg.lr_v, weight_decay=cfg.weight_decay)

    mse = nn.MSELoss()

    diagnostics = {
        "critic_value": [],
    }

    global_step = 0

    for epoch in range(1, cfg.critic_epochs + 1):
        Q1.train(); Q2.train(); V.train()

        q_loss_acc = 0.0
        v_loss_acc = 0.0

        adv_vals = []
        td_vals = []

        pbar = tqdm(loader, desc=f"[IQL] critic/value epoch {epoch}/{cfg.critic_epochs}", leave=False)

        for s, a, r, s_next, done in pbar:
            global_step += 1

            s = s.to(DEVICE)
            a = a.to(DEVICE)
            r = r.to(DEVICE)
            s_next = s_next.to(DEVICE)
            done = done.to(DEVICE)

            # ------------------------------
            # 1) V update (expectile regression towards min(Q1,Q2))
            #    Paper uses dataset actions (s,a) ~ D and learns V(s) such that
            #    Q(s,a) - V(s) fits an expectile.
            # ------------------------------
            with torch.no_grad():
                q1_sa_t = Q1_t(s, a)
                q2_sa_t = Q2_t(s, a)
                q_sa = torch.minimum(q1_sa_t, q2_sa_t)  # clipped double Q

            v_s = V(s)
            diff = q_sa - v_s

            # Optional: clamp advantage for numerical stability (OFF by default)
            if cfg.clamp_adv_min is not None or cfg.clamp_adv_max is not None:
                mn = cfg.clamp_adv_min if cfg.clamp_adv_min is not None else float("-inf")
                mx = cfg.clamp_adv_max if cfg.clamp_adv_max is not None else float("inf")
                diff = diff.clamp(min=mn, max=mx)

            v_loss = expectile_loss(diff, cfg.expectile)

            v_opt.zero_grad(set_to_none=True)
            v_loss.backward()
            nn.utils.clip_grad_norm_(V.parameters(), cfg.grad_clip_norm)
            v_opt.step()

            # ------------------------------
            # 2) Q update with target: r + gamma * V_target(s')
            #    Crucial IQL point: no max over actions at s'.
            # ------------------------------
            with torch.no_grad():
                v_next = V_t(s_next)
                target_q = r + cfg.gamma * (1.0 - done) * v_next

            q1_pred = Q1(s, a)
            q2_pred = Q2(s, a)
            q_loss = mse(q1_pred, target_q) + mse(q2_pred, target_q)

            q_opt.zero_grad(set_to_none=True)
            q_loss.backward()
            nn.utils.clip_grad_norm_(q_params, cfg.grad_clip_norm)
            q_opt.step()

            # ------------------------------
            # 3) EMA updates
            # ------------------------------
            polyak_update(V_t, V, cfg.tau_polyak)
            polyak_update(Q1_t, Q1, cfg.tau_polyak)
            polyak_update(Q2_t, Q2, cfg.tau_polyak)

            q_loss_acc += float(q_loss.item())
            v_loss_acc += float(v_loss.item())

            adv_vals.append((q_sa - v_s).detach().cpu())
            td_vals.append((target_q - torch.minimum(q1_sa_t, q2_sa_t)).detach().cpu())

        # epoch stats
        adv_all = torch.cat(adv_vals)
        td_all = torch.cat(td_vals)

        row = {
            "epoch": epoch,
            "q_loss": q_loss_acc / len(loader),
            "v_loss": v_loss_acc / len(loader),
            "adv_mean": float(adv_all.mean().item()),
            "adv_std": float(adv_all.std().item()),
            "adv_min": float(adv_all.min().item()),
            "adv_max": float(adv_all.max().item()),
            "td_mean": float(td_all.mean().item()),
            "td_std": float(td_all.std().item()),
        }
        diagnostics["critic_value"].append(row)

        if epoch % max(1, cfg.log_every_epochs) == 0:
            print(
                f"[IQL] epoch {epoch:03d} | "
                f"Qloss {row['q_loss']:.6f} | Vloss {row['v_loss']:.6f} | "
                f"Adv std {row['adv_std']:.6f}"
            )

    # Save critic/value checkpoint
    ckpt = {
        "Q1": Q1.state_dict(),
        "Q2": Q2.state_dict(),
        "V": V.state_dict(),
        "Q1_target": Q1_t.state_dict(),
        "Q2_target": Q2_t.state_dict(),
        "V_target": V_t.state_dict(),
        "state_dim": state_dim,
        "num_actions": num_actions,
        "config": asdict(cfg),
    }
    torch.save(ckpt, run_dir / "iql_critics_value.pt")

    with open(run_dir / "diagnostics_critic_value.json", "w") as f:
        json.dump(diagnostics, f, indent=2)

    return Q1, Q2, V, diagnostics


def train_policy_awbc(
    cfg: IQLDiscreteConfig,
    loader,
    Q1: nn.Module,
    Q2: nn.Module,
    V: nn.Module,
    state_dim: int,
    num_actions: int,
    run_dir: Path,
):
    # Freeze Q/V
    Q1.eval(); Q2.eval(); V.eval()
    for p in Q1.parameters(): p.requires_grad_(False)
    for p in Q2.parameters(): p.requires_grad_(False)
    for p in V.parameters(): p.requires_grad_(False)

    pi = PolicyNet(state_dim, num_actions, cfg.hidden1, cfg.hidden2).to(DEVICE)
    opt = optim.Adam(pi.parameters(), lr=cfg.lr_pi)

    diagnostics = {
        "policy": [],
    }

    for epoch in range(1, cfg.policy_epochs + 1):
        pi.train()

        loss_acc = 0.0
        adv_vals = []
        w_vals = []
        clip_count = 0
        total = 0

        pbar = tqdm(loader, desc=f"[IQL] policy epoch {epoch}/{cfg.policy_epochs}", leave=False)
        for s, a, r, s_next, done in pbar:
            s = s.to(DEVICE)
            a = a.to(DEVICE)

            with torch.no_grad():
                q1 = Q1(s, a)
                q2 = Q2(s, a)
                q = torch.minimum(q1, q2)
                v = V(s)
                adv = q - v

                # Optional advantage clamp (OFF by default)
                if cfg.clamp_adv_min is not None or cfg.clamp_adv_max is not None:
                    mn = cfg.clamp_adv_min if cfg.clamp_adv_min is not None else float("-inf")
                    mx = cfg.clamp_adv_max if cfg.clamp_adv_max is not None else float("inf")
                    adv_for_w = adv.clamp(min=mn, max=mx)
                else:
                    adv_for_w = adv

                # Faithful to IQL paper extraction: exp(beta * advantage)
                w_raw = torch.exp(cfg.beta * adv_for_w)

                # clip weights for stability (common in practice)
                clip_count += int((w_raw > cfg.max_weight).sum().item())
                w = w_raw.clamp(max=cfg.max_weight)

                # normalize to keep loss scale stable
                w = w / (w.mean() + 1e-8)

            logits = pi(s)
            logp = torch.log_softmax(logits, dim=1)
            chosen_logp = logp.gather(1, a.unsqueeze(1)).squeeze(1)

            loss = -(w * chosen_logp).mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(pi.parameters(), cfg.policy_grad_clip)
            opt.step()

            loss_acc += float(loss.item())
            adv_vals.append(adv.detach().cpu())
            w_vals.append(w.detach().cpu())
            total += s.shape[0]

        adv_all = torch.cat(adv_vals)
        w_all = torch.cat(w_vals)
        row = {
            "epoch": epoch,
            "loss": loss_acc / len(loader),
            "adv_mean": float(adv_all.mean().item()),
            "adv_std": float(adv_all.std().item()),
            "adv_min": float(adv_all.min().item()),
            "adv_max": float(adv_all.max().item()),
            "w_mean": float(w_all.mean().item()),
            "w_std": float(w_all.std().item()),
            "w_max": float(w_all.max().item()),
            "clip_ratio": float(clip_count / max(1, total)),
        }
        diagnostics["policy"].append(row)

        if epoch % max(1, cfg.log_every_epochs) == 0:
            print(
                f"[IQL] policy epoch {epoch:03d} | "
                f"loss {row['loss']:.6f} | adv std {row['adv_std']:.6f} | "
                f"w std {row['w_std']:.6f} | clip {row['clip_ratio']:.4f}"
            )

    torch.save(
        {
            "policy": pi.state_dict(),
            "state_dim": state_dim,
            "num_actions": num_actions,
            "beta": cfg.beta,
            "max_weight": cfg.max_weight,
            "config": asdict(cfg),
        },
        run_dir / "iql_policy.pt",
    )

    with open(run_dir / "diagnostics_policy.json", "w") as f:
        json.dump(diagnostics, f, indent=2)

    return pi, diagnostics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_dir = create_run_dir(cfg)

    set_seed(cfg.seed)

    print("=" * 80)
    print("IQL DISCRETE (faithful: Double Q + expectile V + AWBC extraction)")
    print("Device:", DEVICE)
    print("Run dir:", run_dir)
    print("Config:", cfg)
    print("=" * 80)

    loader, info = make_dataloader_from_npz(cfg.train_npz, cfg.batch_size)

    print("Dataset:")
    print("  N          :", info["N"])
    print("  state_dim  :", info["state_dim"])
    print("  num_actions:", info["num_actions"])
    print("  r mean/std :", info["reward_mean"], info["reward_std"])
    print("=" * 80)

    # Phase 1: critics + value
    Q1, Q2, V, diag_cv = train_critics_and_value(
        cfg=cfg,
        loader=loader,
        state_dim=info["state_dim"],
        num_actions=info["num_actions"],
        run_dir=run_dir,
    )

    # Phase 2: policy extraction (optional)
    if cfg.train_policy:
        _pi, diag_pi = train_policy_awbc(
            cfg=cfg,
            loader=loader,
            Q1=Q1,
            Q2=Q2,
            V=V,
            state_dim=info["state_dim"],
            num_actions=info["num_actions"],
            run_dir=run_dir,
        )

    print("=" * 80)
    print("Done.")
    print("Saved:")
    print("  - iql_critics_value.pt")
    if cfg.train_policy:
        print("  - iql_policy.pt")
    print("  - diagnostics_*.json")
    print("=" * 80)


if __name__ == "__main__":
    main()