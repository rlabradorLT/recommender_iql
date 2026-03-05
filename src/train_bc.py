import argparse
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from dqn.policiy_network import PolicyNet  # mismo que usas en IQL policy

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# Config (YAML)
# ============================================================

@dataclass
class BCConfig:
    experiment_name: str
    train_npz: str
    output_root: str

    batch_size: int
    epochs: int

    lr: float
    weight_decay: float
    grad_clip_norm: float

    hidden1: int
    hidden2: int

    # Regularización opcional (útil en datasets grandes/sesgados)
    label_smoothing: float  # 0.0 para BC puro
    seed: int


def make_run_dir(output_root: str, experiment_name: str, cfg: dict) -> Path:
    ts = time.strftime("%Y%m%d_%H%M%S")
    cfg_hash = str(abs(hash(json.dumps(cfg, sort_keys=True))))[:6]
    run_dir = Path(output_root) / f"{experiment_name}_{cfg_hash}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    return run_dir


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# Dataset
# ============================================================

def load_dataset(npz_path: str, batch_size: int):
    d = np.load(npz_path)

    obs = torch.tensor(d["observations"], dtype=torch.float32)
    actions = torch.tensor(d["actions"].reshape(-1), dtype=torch.long)

    state_dim = obs.shape[1]
    num_actions = int(d["num_items"])  # tu .npz usa num_items :contentReference[oaicite:4]{index=4}

    dataset = torch.utils.data.TensorDataset(obs, actions)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=(DEVICE == "cuda"),
    )
    return loader, state_dim, num_actions


# ============================================================
# Train BC
# ============================================================

def train_bc(cfg: BCConfig):
    set_seed(cfg.seed)
    run_dir = make_run_dir(cfg.output_root, cfg.experiment_name, asdict(cfg))
    print("Run dir:", run_dir)
    print("Device:", DEVICE)

    loader, state_dim, num_actions = load_dataset(cfg.train_npz, cfg.batch_size)

    # PolicyNet -> logits sobre num_actions (idéntico a tu stack de policy) 
    policy = PolicyNet(state_dim, num_actions, cfg.hidden1, cfg.hidden2).to(DEVICE)

    optimizer = optim.Adam(
        policy.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    # BC oficial: CrossEntropyLoss sobre logits vs acción (máx. log-likelihood)
    # label_smoothing=0.0 => BC “puro”
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

    for epoch in range(1, cfg.epochs + 1):
        policy.train()

        loss_sum = 0.0
        entropy_sum = 0.0
        n_batches = 0

        pbar = tqdm(loader, desc=f"[BC] epoch {epoch}/{cfg.epochs}")
        for s, a in pbar:
            s = s.to(DEVICE, non_blocking=True)
            a = a.to(DEVICE, non_blocking=True)

            logits = policy(s)              # [B, num_actions]
            loss = criterion(logits, a)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.grad_clip_norm)
            optimizer.step()

            # (solo logging)
            with torch.no_grad():
                probs = torch.softmax(logits, dim=1)
                entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()

            loss_sum += float(loss.item())
            entropy_sum += float(entropy.item())
            n_batches += 1

            pbar.set_postfix(loss=f"{loss_sum/n_batches:.5f}", ent=f"{entropy_sum/n_batches:.3f}")

        print(
            f"[BC] epoch {epoch:02d} | "
            f"loss {loss_sum/n_batches:.6f} | "
            f"entropy {entropy_sum/n_batches:.3f}"
        )

    # Guardado ALINEADO con evaluate_models.build_policy_scorer:
    # ckpt["policy"] + state_dim + num_actions 
    ckpt_path = run_dir / "bc_policy.pt"
    torch.save(
        {
            "policy": policy.state_dict(),
            "state_dim": state_dim,
            "num_actions": num_actions,
        },
        ckpt_path,
    )
    print("BC policy saved to:", ckpt_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = BCConfig(**yaml.safe_load(f))

    train_bc(cfg)


if __name__ == "__main__":
    main()