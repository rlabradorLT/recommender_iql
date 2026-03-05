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

from dqn.policiy_network import PolicyNet


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# Config
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

    num_negatives: int
    seed: int


# ============================================================
# Utils
# ============================================================

def make_run_dir(output_root: str, experiment_name: str, cfg: dict):

    ts = time.strftime("%Y%m%d_%H%M%S")
    cfg_hash = str(abs(hash(json.dumps(cfg, sort_keys=True))))[:6]

    run_dir = Path(output_root) / f"{experiment_name}_{cfg_hash}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=False)

    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    return run_dir


def set_seed(seed):

    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# Dataset
# ============================================================

def load_dataset(npz_path):

    d = np.load(npz_path)

    obs = torch.tensor(d["observations"], dtype=torch.float32)
    actions = torch.tensor(d["actions"].reshape(-1), dtype=torch.long)

    state_dim = obs.shape[1]
    num_actions = int(d["num_items"])

    return obs, actions, state_dim, num_actions


# ============================================================
# Negative sampler
# ============================================================

def sample_negatives(batch_size, num_actions, positives, K):

    neg = torch.randint(
        low=0,
        high=num_actions,
        size=(batch_size, K),
        device=positives.device
    )

    mask = neg == positives.unsqueeze(1)

    while mask.any():
        neg[mask] = torch.randint(
            0,
            num_actions,
            (mask.sum(),),
            device=positives.device
        )
        mask = neg == positives.unsqueeze(1)

    return neg


# ============================================================
# Training
# ============================================================

def train(cfg: BCConfig):

    set_seed(cfg.seed)

    run_dir = make_run_dir(cfg.output_root, cfg.experiment_name, asdict(cfg))

    print("Run dir:", run_dir)
    print("Device:", DEVICE)

    obs, actions, state_dim, num_actions = load_dataset(cfg.train_npz)

    dataset = torch.utils.data.TensorDataset(obs, actions)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=(DEVICE == "cuda")
    )

    policy = PolicyNet(
        state_dim,
        num_actions,
        cfg.hidden1,
        cfg.hidden2
    ).to(DEVICE)

    optimizer = optim.Adam(
        policy.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay
    )

    for epoch in range(1, cfg.epochs + 1):

        policy.train()

        loss_sum = 0.0
        n_batches = 0

        pbar = tqdm(loader, desc=f"[BC-Candidate] epoch {epoch}/{cfg.epochs}")

        for s, a in pbar:

            s = s.to(DEVICE)
            a = a.to(DEVICE)

            B = s.size(0)

            neg = sample_negatives(
                B,
                num_actions,
                a,
                cfg.num_negatives
            )

            candidates = torch.cat(
                [a.unsqueeze(1), neg],
                dim=1
            )

            logits = policy(s)

            cand_logits = logits.gather(
                1,
                candidates
            )

            target = torch.zeros(
                B,
                dtype=torch.long,
                device=DEVICE
            )

            loss = nn.functional.cross_entropy(
                cand_logits,
                target
            )

            optimizer.zero_grad(set_to_none=True)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                policy.parameters(),
                cfg.grad_clip_norm
            )

            optimizer.step()

            loss_sum += loss.item()
            n_batches += 1

            pbar.set_postfix(
                loss=f"{loss_sum / n_batches:.5f}"
            )

        print(
            f"[BC-Candidate] epoch {epoch:02d} | "
            f"loss {loss_sum / n_batches:.6f}"
        )

    ckpt_path = run_dir / "bc_candidate.pt"

    torch.save(
        {
            "policy": policy.state_dict(),
            "state_dim": state_dim,
            "num_actions": num_actions
        },
        ckpt_path
    )

    print("Saved:", ckpt_path)


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        required=True
    )

    args = parser.parse_args()

    with open(args.config) as f:

        cfg = BCConfig(
            **yaml.safe_load(f)
        )

    train(cfg)


if __name__ == "__main__":
    main()