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
# CONFIG SYSTEM
# ============================================================

@dataclass
class BCConfig:
    experiment_name: str
    train_npz: str
    output_root: str

    lr: float
    batch_size: int
    epochs: int
    hidden1: int
    hidden2: int
    emb_dim: int


def load_config(path: str):
    with open(path) as f:
        return BCConfig(**yaml.safe_load(f))


def create_run_dir(cfg: BCConfig):
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
# MODEL
# ============================================================

class BCModel(nn.Module):
    def __init__(self, state_dim, num_items, hidden1, hidden2, emb_dim):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, emb_dim),
        )
        self.action_head = nn.Linear(emb_dim, num_items)

    def forward(self, s):
        x = self.backbone(s)
        return self.action_head(x)


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_dir = create_run_dir(cfg)

    print("====================================")
    print("BC TRAINING")
    print("====================================")
    print("Config:", cfg)
    print("Run dir:", run_dir)
    print("Device:", DEVICE)
    print("====================================")

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    d = np.load(cfg.train_npz)

    obs = torch.tensor(d["observations"], dtype=torch.float32)
    actions = torch.tensor(d["actions"].flatten(), dtype=torch.long)

    N, state_dim = obs.shape
    num_items = int(actions.max()) + 1

    print("Transitions:", N)
    print("State dim:", state_dim)
    print("Num items:", num_items)

    dataset = torch.utils.data.TensorDataset(obs, actions)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = BCModel(
        state_dim=state_dim,
        num_items=num_items,
        hidden1=cfg.hidden1,
        hidden2=cfg.hidden2,
        emb_dim=cfg.emb_dim
    ).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = nn.CrossEntropyLoss()

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_loss = 0.0

        for s, a in tqdm(loader, desc=f"Epoch {epoch}/{cfg.epochs}", leave=False):
            s = s.to(DEVICE)
            a = a.to(DEVICE)

            logits = model(s)
            loss = criterion(logits, a)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())

        print(f"Epoch {epoch:02d} | Loss: {total_loss/len(loader):.4f}")

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "state_dim": state_dim,
            "num_items": num_items,
            "hidden1": cfg.hidden1,
            "hidden2": cfg.hidden2,
            "emb_dim": cfg.emb_dim,
        },
        run_dir / "bc_model.pt"
    )

    print("BC training complete.")
    print("Saved to:", run_dir / "bc_model.pt")


if __name__ == "__main__":
    main()