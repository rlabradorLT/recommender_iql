import argparse
import json
from pathlib import Path

import yaml
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# METRICS (streaming)
# ============================================================

@torch.no_grad()
def metrics_from_scores_stream(score_fn, obs, candidates, eval_batch: int, ks=(5, 10, 20)):
    sums = {f"HR@{k}": 0.0 for k in ks}
    sums.update({f"NDCG@{k}": 0.0 for k in ks})
    n_total = 0

    for i in tqdm(range(0, obs.shape[0], eval_batch), desc="Eval", leave=False):
        ob = obs[i:i + eval_batch]
        cb = candidates[i:i + eval_batch]

        scores = score_fn(ob, cb)
        pos_scores = scores[:, 0:1]
        better = (scores > pos_scores).sum(dim=1)
        rank = better + 1

        for k in ks:
            hit = (rank <= k).float()
            sums[f"HR@{k}"] += float(hit.sum().item())
            sums[f"NDCG@{k}"] += float((hit / torch.log2(rank.float() + 1.0)).sum().item())

        n_total += scores.shape[0]

    return {k: v / n_total for k, v in sums.items()}


# ============================================================
# LOADERS
# ============================================================

def load_split(data_npz, candidates_path):
    d = np.load(data_npz)
    obs = torch.tensor(d["observations"], dtype=torch.float32)
    candidates = torch.tensor(np.load(candidates_path), dtype=torch.long)
    return obs, candidates


def filter_oob(obs: torch.Tensor, candidates: torch.Tensor, num_items: int, tag: str):
    cmin = int(candidates.min().item())
    cmax = int(candidates.max().item())

    if cmin < 0:
        raise ValueError(f"{tag}: negative candidate ids (min={cmin})")

    valid_mask = (candidates < num_items).all(dim=1)
    dropped = int((~valid_mask).sum().item())

    if dropped > 0:
        print(
            f"{tag}: dropping {dropped} / {candidates.shape[0]} rows "
            f"(candidate id out of bounds, max={cmax}, num_items={num_items})"
        )
        obs = obs[valid_mask]
        candidates = candidates[valid_mask]

    if candidates.numel():
        cmax2 = int(candidates.max().item())
        if cmax2 >= num_items:
            raise ValueError(f"{tag}: still OOB after filtering")

    return obs, candidates


# ============================================================
# CONFIG
# ============================================================

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)

    dataset_dir = Path(cfg["dataset_dir"])
    split = cfg.get("split", "val")
    eval_batch = cfg.get("eval_batch", 2048)

    obs, candidates = load_split(
        dataset_dir / f"rl_{split}.npz",
        dataset_dir / f"{split}_cands_rl.npy",
    )

    results = {}

    # --------------------------------------------------------
    # BC
    # --------------------------------------------------------
    bc_ckpt = cfg.get("bc_ckpt")

    if bc_ckpt:

        print("Evaluating BC...")
        ckpt = torch.load(bc_ckpt, map_location="cpu")

        state_dim = int(ckpt["state_dim"])
        num_items = int(ckpt["num_items"])
        hidden1 = int(ckpt.get("hidden1", 256))
        hidden2 = int(ckpt.get("hidden2", 128))
        emb_dim = int(ckpt.get("emb_dim", 64))

        obs_bc, cand_bc = filter_oob(obs, candidates, num_items, "BC")

        class BCModel(nn.Module):
            def __init__(self):
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
                return self.action_head(self.backbone(s))

        model = BCModel().to(DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        @torch.no_grad()
        def bc_score_fn(ob, cb):
            ob = ob.to(DEVICE)
            cb = cb.to(DEVICE)
            logits = model(ob)
            return torch.gather(logits, 1, cb).float().cpu()

        results["BC"] = metrics_from_scores_stream(
            bc_score_fn, obs_bc, cand_bc, eval_batch
        )

    # --------------------------------------------------------
    # IQL
    # --------------------------------------------------------
    iql_ckpt = cfg.get("iql_ckpt")

    if iql_ckpt:

        print("Evaluating IQL...")
        ckpt = torch.load(iql_ckpt, map_location="cpu")

        state_dim = int(ckpt["state_dim"])
        num_items = int(ckpt["num_items"])

        obs_iql, cand_iql = filter_oob(obs, candidates, num_items, "IQL")

        class QNetwork(nn.Module):
            def __init__(self):
                super().__init__()
                emb_dim = 64
                hidden1 = 256
                hidden2 = 128

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

        Q = QNetwork().to(DEVICE)
        Q.load_state_dict(ckpt["Q"])
        Q.eval()

        @torch.no_grad()
        def q_score_fn(ob, cb):

            ob = ob.to(DEVICE)
            cb = cb.to(DEVICE)

            B, C = cb.shape
            out = torch.empty((B, C), device="cpu")

            cand_chunk = 25

            for j in range(0, C, cand_chunk):

                cb_j = cb[:, j:j + cand_chunk]
                c = cb_j.shape[1]

                ob_rep = ob.unsqueeze(1).expand(-1, c, -1).reshape(B * c, -1)
                a_flat = cb_j.reshape(B * c)

                q_flat = Q(ob_rep, a_flat).reshape(B, c)
                out[:, j:j + c] = q_flat.float().cpu()

            return out

        results["IQL"] = metrics_from_scores_stream(
            q_score_fn, obs_iql, cand_iql, eval_batch
        )

    # --------------------------------------------------------
    # AWR
    # --------------------------------------------------------
    awr_ckpt = cfg.get("awr_ckpt")

    if awr_ckpt:

        print("Evaluating AWR...")
        ckpt = torch.load(awr_ckpt, map_location="cpu")

        state_dim = int(ckpt["state_dim"])
        num_items = int(ckpt["num_items"])

        obs_awr, cand_awr = filter_oob(obs, candidates, num_items, "AWR")

        class PolicyNet(nn.Module):
            def __init__(self):
                super().__init__()

                hidden1 = 256
                hidden2 = 128

                self.net = nn.Sequential(
                    nn.Linear(state_dim, hidden1),
                    nn.ReLU(),
                    nn.Linear(hidden1, hidden2),
                    nn.ReLU(),
                    nn.Linear(hidden2, num_items),
                )

            def forward(self, s):
                return self.net(s)

        pi = PolicyNet().to(DEVICE)
        pi.load_state_dict(ckpt["policy"])
        pi.eval()

        @torch.no_grad()
        def awr_score_fn(ob, cb):
            ob = ob.to(DEVICE)
            cb = cb.to(DEVICE)
            logits = pi(ob)
            return torch.gather(logits, 1, cb).float().cpu()

        results["AWR"] = metrics_from_scores_stream(
            awr_score_fn, obs_awr, cand_awr, eval_batch
        )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    print("\n====================================")
    print(f"RESULTS ({split})")
    print("====================================")

    for model_name, metrics in results.items():
        print(model_name)
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
        print()

    out_path = dataset_dir / f"metrics_{split}.json"

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print("Saved metrics to:", out_path)


if __name__ == "__main__":
    main()