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
class AWRConfig:
    experiment_name: str

    train_npz: str
    iql_ckpt: str
    output_root: str

    beta: float
    max_weight: float

    epochs: int
    batch_size: int
    lr: float
    grad_clip: float

    emb_dim: int


def load_config(path: str):
    with open(path) as f:
        return AWRConfig(**yaml.safe_load(f))


def create_run_dir(cfg: AWRConfig):
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
    def __init__(self, num_items: int, state_dim: int, emb_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(num_items + 1, emb_dim)
        self.net = nn.Sequential(
            nn.Linear(state_dim + emb_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, s, a):
        a_emb = self.embedding(a)
        x = torch.cat([s, a_emb], dim=1)
        return self.net(x).squeeze(-1)


class VNetwork(nn.Module):
    def __init__(self, state_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, s):
        return self.net(s).squeeze(-1)


class PolicyNet(nn.Module):
    def __init__(self, state_dim: int, num_items: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_items),
        )

    def forward(self, s):
        return self.net(s)


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_dir = create_run_dir(cfg)

    print("=" * 60)
    print("AWR POLICY TRAINING (IQL extraction with diagnostics)")
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
    actions = torch.tensor(d["actions"].reshape(-1), dtype=torch.long)

    N, state_dim = obs.shape
    num_items = int(actions.max()) + 1

    print("Transitions:", N)
    print("State dim:", state_dim)
    print("Num items:", num_items)

    # --------------------------------------------------------
    # Load IQL checkpoint
    # --------------------------------------------------------

    ckpt = torch.load(cfg.iql_ckpt, map_location="cpu")

    Q = QNetwork(num_items, state_dim, cfg.emb_dim).to(DEVICE)
    V = VNetwork(state_dim).to(DEVICE)

    Q.load_state_dict(ckpt["Q"])
    V.load_state_dict(ckpt["V"])

    Q.eval(); V.eval()
    for p in Q.parameters(): p.requires_grad_(False)
    for p in V.parameters(): p.requires_grad_(False)

    # --------------------------------------------------------
    # Policy
    # --------------------------------------------------------

    pi = PolicyNet(state_dim, num_items).to(DEVICE)
    opt = optim.Adam(pi.parameters(), lr=cfg.lr)

    dataset = torch.utils.data.TensorDataset(obs, actions)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True
    )

    metrics_log = []

    # ============================================================
    # TRAINING LOOP
    # ============================================================

    for epoch in range(1, cfg.epochs + 1):

        pi.train()

        epoch_loss = 0.0

        adv_all = []
        w_all = []
        clip_count = 0
        total_samples = 0

        for s, a in tqdm(loader, desc=f"Epoch {epoch}/{cfg.epochs}", leave=False):

            s = s.to(DEVICE)
            a = a.to(DEVICE)

            with torch.no_grad():
                q = Q(s, a)
                v = V(s)
                adv = q - v

                w_raw = torch.exp(adv / cfg.beta)
                clip_count += (w_raw > cfg.max_weight).sum().item()

                w = w_raw.clamp(max=cfg.max_weight)
                w = w / (w.mean() + 1e-8)

            logits = pi(s)
            logp = torch.log_softmax(logits, dim=1)
            chosen_logp = logp.gather(1, a.unsqueeze(1)).squeeze(1)

            loss = -(w * chosen_logp).mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(pi.parameters(), cfg.grad_clip)
            opt.step()

            epoch_loss += loss.item()

            adv_all.append(adv.detach().cpu())
            w_all.append(w.detach().cpu())
            total_samples += len(s)

        # --------------------------------------------------------
        # Aggregate diagnostics
        # --------------------------------------------------------

        adv_all = torch.cat(adv_all)
        w_all = torch.cat(w_all)

        adv_mean = adv_all.mean().item()
        adv_std = adv_all.std().item()
        adv_min = adv_all.min().item()
        adv_max = adv_all.max().item()

        w_mean = w_all.mean().item()
        w_std = w_all.std().item()
        w_max = w_all.max().item()

        clip_ratio = clip_count / total_samples

        print(
            f"Epoch {epoch:02d} | "
            f"loss: {epoch_loss/len(loader):.4f} | "
            f"adv_std: {adv_std:.6f} | "
            f"w_std: {w_std:.6f} | "
            f"clip%: {clip_ratio:.4f}"
        )

        metrics_log.append({
            "epoch": epoch,
            "loss": epoch_loss/len(loader),
            "adv_mean": adv_mean,
            "adv_std": adv_std,
            "adv_min": adv_min,
            "adv_max": adv_max,
            "w_mean": w_mean,
            "w_std": w_std,
            "w_max": w_max,
            "clip_ratio": clip_ratio
        })

    # --------------------------------------------------------
    # Save diagnostics
    # --------------------------------------------------------

    with open(run_dir / "diagnostics.json", "w") as f:
        json.dump(metrics_log, f, indent=2)

    torch.save(
        {
            "policy": pi.state_dict(),
            "state_dim": state_dim,
            "num_items": num_items,
            "beta": cfg.beta,
            "max_weight": cfg.max_weight,
            "iql_ckpt": cfg.iql_ckpt,
        },
        run_dir / "policy.pt"
    )
    print("adv mean:", adv.mean().item())
    print("adv std :", adv.std().item())
    print("weight mean:", w.mean().item())
    print("weight max :", w.max().item())
    
    print("=" * 60)
    print("Training complete.")
    print("Saved policy to:", run_dir / "policy.pt")
    print("Saved diagnostics to:", run_dir / "diagnostics.json")
    print("=" * 60)


if __name__ == "__main__":
    main()