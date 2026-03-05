# train_cql_discrete_stable.py
# ------------------------------------------------------------
# Conservative Q-Learning (CQL) - DISCRETE, OFFLINE, LARGE ACTION SPACE
#
# Key stability features for recommender-style action spaces (|A| large):
#  1) Double Q + Polyak target nets
#  2) TD target uses min(Q1_t, Q2_t) and greedy action (DQN-like)
#  3) CQL(H) conservative penalty with exact logsumexp over actions
#  4) IMPORTANT: "Centered" CQL penalty subtracting T*log|A|
#     This removes the constant offset induced purely by catalog size.
#  5) Penalty computed on Q_min = min(Q1, Q2) to reduce variance
#  6) Huber TD + grad clipping
# ------------------------------------------------------------

import argparse
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

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


def safe_logsumexp(q_all: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    temperature * logsumexp(q_all / temperature) over actions
    q_all: [B, A] -> returns [B]
    """
    if temperature <= 0:
        raise ValueError("cql_temperature must be > 0")
    return temperature * torch.logsumexp(q_all / temperature, dim=1)


def cql_centered_penalty(
    q_all: torch.Tensor,
    a_data: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """
    Centered CQL(H) penalty per sample:
        (T*logsumexp(Q/T) - T*log|A|) - Q(s,a_data)

    Why centered:
      For large |A|, T*logsumexp(Q/T) ≈ T*log|A| when Q ~ 0,
      which adds a huge constant unrelated to OOD-ness.
      Subtracting T*log|A| removes that constant baseline.

    q_all: [B, A]
    a_data: [B]
    returns: [B]
    """
    b, num_actions = q_all.shape
    q_data = q_all.gather(1, a_data.unsqueeze(1)).squeeze(1)  # [B]

    lse = safe_logsumexp(q_all, temperature=temperature)      # [B]

    # baseline = T * log|A| (computed on device, stable)
    baseline = temperature * torch.log(torch.tensor(float(num_actions), device=q_all.device))

    return (lse - baseline) - q_data


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
    alpha_cql: float
    cql_temperature: float

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
    max_q_abs: Optional[float]

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

    # target critics
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

    global_step = 0
    diag = {"epochs": []}

    alpha = float(cfg.alpha_cql)
    temperature = float(cfg.cql_temperature)

    for epoch in range(1, cfg.epochs + 1):
        Q1.train()
        Q2.train()

        td_loss_sum = 0.0
        cql_loss_sum = 0.0
        total_loss_sum = 0.0

        # epoch stats
        penalty_means = []
        qdata_means = []
        lse_means = []

        pbar = tqdm(loader, desc=f"[CQL-stable] epoch {epoch}/{cfg.epochs}", leave=False)
        for s, a, r, s_next, done in pbar:
            global_step += 1

            s = s.to(DEVICE)
            a = a.to(DEVICE)
            r = r.to(DEVICE)
            s_next = s_next.to(DEVICE)
            done = done.to(DEVICE)

            # ------------------------------------------------------------
            # (1) TD target: DQN-like with clipped double Q target critics
            # ------------------------------------------------------------
            with torch.no_grad():
                q1_next_all = Q1_t(s_next)   # [B,A]
                q2_next_all = Q2_t(s_next)   # [B,A]
                q_next_min = torch.minimum(q1_next_all, q2_next_all)

                a_next = torch.argmax(q_next_min, dim=1)  # [B]
                q_next = q_next_min.gather(1, a_next.unsqueeze(1)).squeeze(1)  # [B]

                target = r + cfg.gamma * (1.0 - done) * q_next

            # ------------------------------------------------------------
            # (2) Current Q predictions
            # ------------------------------------------------------------
            q1_all = Q1(s)  # [B,A]
            q2_all = Q2(s)  # [B,A]

            if cfg.max_q_abs is not None:
                q1_all = q1_all.clamp(-cfg.max_q_abs, cfg.max_q_abs)
                q2_all = q2_all.clamp(-cfg.max_q_abs, cfg.max_q_abs)

            q1_sa = q1_all.gather(1, a.unsqueeze(1)).squeeze(1)  # [B]
            q2_sa = q2_all.gather(1, a.unsqueeze(1)).squeeze(1)  # [B]

            # TD loss (Huber) both critics
            td_loss = (
                F.huber_loss(q1_sa, target, delta=cfg.huber_delta) +
                F.huber_loss(q2_sa, target, delta=cfg.huber_delta)
            )

            # ------------------------------------------------------------
            # (3) CQL(H) centered penalty on Q_min (more stable in recsys)
            # ------------------------------------------------------------
            q_min_all = torch.minimum(q1_all, q2_all)  # [B,A]
            penalty_vec = cql_centered_penalty(q_min_all, a, temperature=temperature)  # [B]
            cql_loss = alpha * penalty_vec.mean()

            # Total loss
            loss = td_loss + cql_loss

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
            # stats
            # ------------------------------------------------------------
            td_loss_sum += float(td_loss.item())
            cql_loss_sum += float(cql_loss.item())
            total_loss_sum += float(loss.item())

            with torch.no_grad():
                # report uncentered LSE and Qdata for intuition
                lse_unc = safe_logsumexp(q_min_all, temperature=temperature).mean()
                qdata = q_min_all.gather(1, a.unsqueeze(1)).squeeze(1).mean()
                penalty_means.append(penalty_vec.mean().detach().cpu())
                qdata_means.append(qdata.detach().cpu())
                lse_means.append(lse_unc.detach().cpu())

        # epoch summary
        penalty_mean = float(torch.stack(penalty_means).mean().item())
        qdata_mean = float(torch.stack(qdata_means).mean().item())
        lse_mean = float(torch.stack(lse_means).mean().item())

        row = {
            "epoch": epoch,
            "td_loss": td_loss_sum / len(loader),
            "cql_loss": cql_loss_sum / len(loader),
            "total_loss": total_loss_sum / len(loader),
            "penalty_centered_mean": penalty_mean,
            "qdata_mean": qdata_mean,
            "lse_uncentered_mean": lse_mean,
            "alpha": alpha,
            "temperature": temperature,
        }
        diag["epochs"].append(row)

        if epoch % max(1, cfg.log_every_epochs) == 0:
            print(
                f"[CQL-stable] epoch {epoch:03d} | "
                f"TD {row['td_loss']:.6f} | CQL {row['cql_loss']:.6f} | "
                f"Penalty(centered) {row['penalty_centered_mean']:.4f} | "
                f"Qdata {row['qdata_mean']:.3f} | "
                f"LSE(unc) {row['lse_uncentered_mean']:.3f} | "
                f"alpha {row['alpha']:.4f}"
            )

        # fail-fast
        if not np.isfinite(row["total_loss"]) or not np.isfinite(row["td_loss"]) or not np.isfinite(row["cql_loss"]):
            raise RuntimeError("Divergence detected (NaN/Inf). Check reward_scale/gamma/lr/alpha and dataset sanity.")

    # save
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
    print("CQL DISCRETE STABLE (offline, large action space) - CENTERED PENALTY")
    print("Device:", DEVICE)
    print("Config:", cfg)
    print("=" * 80)

    train(cfg)


if __name__ == "__main__":
    main()