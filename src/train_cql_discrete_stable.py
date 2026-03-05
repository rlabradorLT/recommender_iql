# train_cql_discrete_stable.py
# ------------------------------------------------------------
# Conservative Q-Learning (CQL) for OFFLINE RL with DISCRETE actions
# Designed for large action spaces (e.g., recsys items) and stability.
#
# Aligns with CQL(H) in the paper (discrete setting): add conservative penalty
#   alpha * ( logsumexp_a Q(s,a) - Q(s,a_data) )
# on top of a standard TD (Bellman) loss.
#
# Notes:
# - Uses Double Q (two critics) + Polyak target networks.
# - Uses "min(Q1, Q2)" for the bootstrap target (clipped double Q).
# - Uses Huber loss + grad clipping.
# - YAML config driven (similar to your IQL script).
# ------------------------------------------------------------

import argparse
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from tqdm import tqdm

from dqn.q_network import QNetwork

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


def make_run_dir(output_root: str, experiment_name: str, cfg: dict) -> Path:
    ts = time.strftime("%Y%m%d_%H%M%S")
    cfg_hash = str(abs(hash(json.dumps(cfg, sort_keys=True))))[:6]
    run_dir = Path(output_root) / f"{experiment_name}_{cfg_hash}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    return run_dir


def safe_logsumexp(q_all: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Computes: temperature * logsumexp(q_all / temperature)
    Stable for large action spaces.
    Input: q_all shape [B, A]
    Output: shape [B]
    """
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    x = q_all / temperature
    # torch.logsumexp is stable
    return temperature * torch.logsumexp(x, dim=1)


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

    # CQL
    alpha_cql: float                 # conservative penalty weight (fixed)
    cql_temperature: float           # temp for logsumexp; 1.0 typical
    cql_variant: str                 # "H" (CQL(H)) or "simple" (same in discrete exact LSE)
    # optional Lagrange (auto alpha) - kept simple/off by default
    use_lagrange: bool
    target_action_gap: float         # if use_lagrange, target for (logsumexp - Qdata)
    lr_alpha: float                  # alpha optimizer LR if use_lagrange
    alpha_min: float
    alpha_max: float

    # training
    epochs: int
    batch_size: int
    lr_q: float
    weight_decay: float
    grad_clip_norm: float

    # target nets
    tau_polyak: float
    target_update_period: int

    # model
    hidden1: int
    hidden2: int

    # stability knobs
    huber_delta: float
    max_q_abs: Optional[float]       # optional clamp for Q predictions (diagnostic stability)

    # logging
    log_every_epochs: int


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

    if actions.min().item() < 0 or actions.max().item() >= num_actions:
        raise ValueError(
            f"actions out of range: min={actions.min().item()} "
            f"max={actions.max().item()} num_actions={num_actions}"
        )

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
# CQL Loss (discrete exact)
# ============================================================

def cql_penalty(
    q_all: torch.Tensor,
    a_data: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """
    CQL(H) penalty term (per sample):
      logsumexp_a Q(s,a) - Q(s,a_data)
    Using exact logsumexp over discrete actions.
    q_all: [B, A]
    a_data: [B]
    returns: [B]
    """
    q_data = q_all.gather(1, a_data.unsqueeze(1)).squeeze(1)     # [B]
    lse = safe_logsumexp(q_all, temperature=temperature)         # [B]
    return lse - q_data


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

    # targets
    Q1_t = QNetwork(state_dim, num_actions, cfg.hidden1, cfg.hidden2).to(DEVICE)
    Q2_t = QNetwork(state_dim, num_actions, cfg.hidden1, cfg.hidden2).to(DEVICE)
    Q1_t.load_state_dict(Q1.state_dict())
    Q2_t.load_state_dict(Q2.state_dict())
    Q1_t.eval()
    Q2_t.eval()

    q_opt = optim.Adam(
        list(Q1.parameters()) + list(Q2.parameters()),
        lr=cfg.lr_q,
        weight_decay=cfg.weight_decay,
        eps=1e-8,
    )

    # Optional Lagrange alpha tuning (kept stable by clamping)
    # In continuous CQL they use Lagrangian to tune alpha. In discrete it's also possible.
    # Here we tune alpha to match target_action_gap on the penalty mean.
    if cfg.use_lagrange:
        log_alpha = torch.tensor(np.log(max(cfg.alpha_cql, 1e-6)), device=DEVICE, requires_grad=True)
        alpha_opt = optim.Adam([log_alpha], lr=cfg.lr_alpha, eps=1e-8)
    else:
        log_alpha = None
        alpha_opt = None

    global_step = 0
    diag = {"epochs": []}

    for epoch in range(1, cfg.epochs + 1):
        Q1.train()
        Q2.train()

        td_loss_sum = 0.0
        cql_loss_sum = 0.0
        total_loss_sum = 0.0

        stats = {
            "q_data_mean": [],
            "q_lse_mean": [],
            "penalty_mean": [],
            "target_mean": [],
            "td_mean": [],
        }

        pbar = tqdm(loader, desc=f"[CQL-stable] epoch {epoch}/{cfg.epochs}", leave=False)
        for s, a, r, s_next, done in pbar:
            global_step += 1

            s = s.to(DEVICE)
            a = a.to(DEVICE)
            r = r.to(DEVICE)
            s_next = s_next.to(DEVICE)
            done = done.to(DEVICE)

            # ------------------------------------------------------------
            # (1) Compute bootstrap target (clipped double Q + greedy)
            # Standard DQN-like target but using min(Q1,Q2) for stability.
            # ------------------------------------------------------------
            with torch.no_grad():
                q1_next_all = Q1_t(s_next)              # [B,A]
                q2_next_all = Q2_t(s_next)              # [B,A]
                q_next_min = torch.minimum(q1_next_all, q2_next_all)

                # greedy action under the MIN critics (stable)
                a_next = torch.argmax(q_next_min, dim=1)  # [B]
                q_next = q_next_min.gather(1, a_next.unsqueeze(1)).squeeze(1)

                target = r + cfg.gamma * (1.0 - done) * q_next

            # ------------------------------------------------------------
            # (2) TD loss for both critics
            # ------------------------------------------------------------
            q1_all = Q1(s)  # [B,A]
            q2_all = Q2(s)  # [B,A]

            if cfg.max_q_abs is not None:
                q1_all = q1_all.clamp(-cfg.max_q_abs, cfg.max_q_abs)
                q2_all = q2_all.clamp(-cfg.max_q_abs, cfg.max_q_abs)

            q1_sa = q1_all.gather(1, a.unsqueeze(1)).squeeze(1)  # [B]
            q2_sa = q2_all.gather(1, a.unsqueeze(1)).squeeze(1)  # [B]

            td_loss = (
                F.huber_loss(q1_sa, target, delta=cfg.huber_delta) +
                F.huber_loss(q2_sa, target, delta=cfg.huber_delta)
            )

            # ------------------------------------------------------------
            # (3) CQL conservative penalty
            # CQL(H): logsumexp(Q) - Q_data
            # We apply it to each critic separately and sum.
            # ------------------------------------------------------------
            pen1 = cql_penalty(q1_all, a, temperature=cfg.cql_temperature)  # [B]
            pen2 = cql_penalty(q2_all, a, temperature=cfg.cql_temperature)  # [B]
            penalty = 0.5 * (pen1 + pen2)                                  # [B]

            # Alpha handling
            if cfg.use_lagrange:
                # alpha = exp(log_alpha), clamped
                alpha = torch.exp(log_alpha).clamp(cfg.alpha_min, cfg.alpha_max)

                # In CQL Lagrange form: alpha * (penalty - target_gap)
                # We update alpha to push penalty_mean toward target_action_gap.
                cql_term = alpha * (penalty.mean() - cfg.target_action_gap)

                # Update alpha (gradient ASCENT on alpha objective usually; implement via minimizing negative)
                alpha_opt.zero_grad(set_to_none=True)
                alpha_loss = -(alpha * (penalty.mean() - cfg.target_action_gap)).detach()
                # Note: detach here to avoid second-order effects; stable in practice.
                # If you want exact gradients: remove detach and compute alpha loss separately.
                # But for offline stability, this is often preferred.
                alpha_loss.backward()
                alpha_opt.step()

                # For critic loss we still use current alpha
                alpha = torch.exp(log_alpha).clamp(cfg.alpha_min, cfg.alpha_max).detach()
            else:
                alpha = torch.tensor(float(cfg.alpha_cql), device=DEVICE)
                cql_term = alpha * penalty.mean()

            # Total loss
            loss = td_loss + cql_term

            q_opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(list(Q1.parameters()) + list(Q2.parameters()), cfg.grad_clip_norm)
            q_opt.step()

            # ------------------------------------------------------------
            # (4) Target update (slow + periodic)
            # ------------------------------------------------------------
            if (global_step % int(cfg.target_update_period)) == 0:
                polyak_update(Q1_t, Q1, cfg.tau_polyak)
                polyak_update(Q2_t, Q2, cfg.tau_polyak)

            # ------------------------------------------------------------
            # Stats
            # ------------------------------------------------------------
            td_loss_sum += float(td_loss.item())
            cql_loss_sum += float(cql_term.item())
            total_loss_sum += float(loss.item())

            with torch.no_grad():
                q_data_mean = 0.5 * (q1_sa.mean() + q2_sa.mean())
                q_lse_mean = 0.5 * (safe_logsumexp(q1_all, cfg.cql_temperature).mean() +
                                    safe_logsumexp(q2_all, cfg.cql_temperature).mean())
                stats["q_data_mean"].append(q_data_mean.detach().cpu())
                stats["q_lse_mean"].append(q_lse_mean.detach().cpu())
                stats["penalty_mean"].append(penalty.mean().detach().cpu())
                stats["target_mean"].append(target.mean().detach().cpu())
                stats["td_mean"].append((target - 0.5 * (q1_sa + q2_sa)).mean().detach().cpu())

        # epoch summary
        q_data_mean = float(torch.stack(stats["q_data_mean"]).mean().item())
        q_lse_mean = float(torch.stack(stats["q_lse_mean"]).mean().item())
        penalty_mean = float(torch.stack(stats["penalty_mean"]).mean().item())
        target_mean = float(torch.stack(stats["target_mean"]).mean().item())
        td_mean = float(torch.stack(stats["td_mean"]).mean().item())

        row = {
            "epoch": epoch,
            "td_loss": td_loss_sum / len(loader),
            "cql_loss": cql_loss_sum / len(loader),
            "total_loss": total_loss_sum / len(loader),
            "q_data_mean": q_data_mean,
            "q_lse_mean": q_lse_mean,
            "penalty_mean": penalty_mean,
            "target_mean": target_mean,
            "td_error_mean": td_mean,
            "alpha": float(torch.exp(log_alpha).clamp(cfg.alpha_min, cfg.alpha_max).item()) if cfg.use_lagrange else float(cfg.alpha_cql),
        }
        diag["epochs"].append(row)

        if epoch % max(1, cfg.log_every_epochs) == 0:
            print(
                f"[CQL-stable] epoch {epoch:03d} | "
                f"TD {row['td_loss']:.6f} | CQL {row['cql_loss']:.6f} | "
                f"Penalty {row['penalty_mean']:.4f} | "
                f"Qdata {row['q_data_mean']:.3f} | LSE {row['q_lse_mean']:.3f} | "
                f"alpha {row['alpha']:.4f}"
            )

        # Fail-fast on divergence
        if not np.isfinite(row["total_loss"]) or not np.isfinite(row["td_loss"]) or not np.isfinite(row["cql_loss"]):
            raise RuntimeError("Divergence detected (NaN/Inf). Check reward_scale/gamma/lr/alpha and dataset sanity.")

    # Save
    torch.save(
        {
            "Q1": Q1.state_dict(),
            "Q2": Q2.state_dict(),
            "Q1_target": Q1_t.state_dict(),
            "Q2_target": Q2_t.state_dict(),
            "state_dim": state_dim,
            "num_actions": num_actions,
            "config": asdict(cfg),
        },
        run_dir / "cql_discrete_stable.pt",
    )
    (run_dir / "diagnostics.json").write_text(json.dumps(diag, indent=2))

    print("Saved:", run_dir / "cql_discrete_stable.pt")
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
    print("CQL DISCRETE STABLE (offline, large action space)")
    print("Device:", DEVICE)
    print("Config:", cfg)
    print("=" * 80)

    train(cfg)


if __name__ == "__main__":
    main()