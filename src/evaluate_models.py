import argparse
import json
from pathlib import Path

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
    """
    score_fn(obs_batch, cand_batch) -> scores [B, C]
    Computes HR/NDCG in streaming way to avoid keeping all scores in memory.
    """
    sums = {f"HR@{k}": 0.0 for k in ks}
    sums.update({f"NDCG@{k}": 0.0 for k in ks})
    n_total = 0

    for i in tqdm(range(0, obs.shape[0], eval_batch), desc="Eval", leave=False):
        ob = obs[i:i + eval_batch]
        cb = candidates[i:i + eval_batch]

        scores = score_fn(ob, cb)  # [B, C]
        pos_scores = scores[:, 0:1]
        better = (scores > pos_scores).sum(dim=1)
        rank = better + 1  # [B]

        for k in ks:
            hit = (rank <= k).float()
            sums[f"HR@{k}"] += float(hit.sum().item())
            sums[f"NDCG@{k}"] += float((hit / torch.log2(rank.float() + 1.0)).sum().item())

        n_total += scores.shape[0]

    out = {k: v / n_total for k, v in sums.items()}
    return out


# ============================================================
# LOADERS
# ============================================================

def load_split(data_npz, candidates_path):
    d = np.load(data_npz)
    obs = torch.tensor(d["observations"], dtype=torch.float32)
    candidates = torch.tensor(np.load(candidates_path), dtype=torch.long)
    return obs, candidates


def filter_oob(obs: torch.Tensor, candidates: torch.Tensor, num_items: int, tag: str):
    """
    Keep only rows where all candidate ids are within [0, num_items-1].
    """
    cmin = int(candidates.min().item())
    cmax = int(candidates.max().item())
    if cmin < 0:
        raise ValueError(f"{tag}: negative candidate ids (min={cmin})")

    valid_mask = (candidates < num_items).all(dim=1)
    dropped = int((~valid_mask).sum().item())
    if dropped > 0:
        print(f"{tag}: dropping {dropped} / {candidates.shape[0]} rows "
              f"(candidate id out of bounds, max={cmax}, num_items={num_items})")
        obs = obs[valid_mask]
        candidates = candidates[valid_mask]

    # final sanity
    if candidates.numel():
        cmax2 = int(candidates.max().item())
        if cmax2 >= num_items:
            raise ValueError(f"{tag}: still OOB after filtering (max={cmax2}, num_items={num_items})")
    return obs, candidates


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--bc_ckpt", default=None)
    parser.add_argument("--iql_ckpt", default=None)
    parser.add_argument("--awr_ckpt", default=None)
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--eval_batch", type=int, default=2048)  # reduce if OOM
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)

    obs, candidates = load_split(
        dataset_dir / f"rl_{args.split}.npz",
        dataset_dir / f"{args.split}_cands_rl.npy",
    )

    results = {}

    # --------------------------------------------------------
    # BC
    # --------------------------------------------------------
    if args.bc_ckpt:
        print("Evaluating BC...")
        ckpt = torch.load(args.bc_ckpt, map_location="cpu")
        state_dim = int(ckpt["state_dim"])
        num_items = int(ckpt["num_items"])
        hidden1 = int(ckpt.get("hidden1", 256))
        hidden2 = int(ckpt.get("hidden2", 128))
        emb_dim = int(ckpt.get("emb_dim", 64))

        obs_bc, cand_bc = filter_oob(obs, candidates, num_items=num_items, tag="BC")

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

        model = BCModel(state_dim, num_items, hidden1, hidden2, emb_dim).to(DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        @torch.no_grad()
        def bc_score_fn(ob, cb):
            ob = ob.to(DEVICE, non_blocking=True)
            cb = cb.to(DEVICE, non_blocking=True)
            logits = model(ob)  # [B, num_items]
            return torch.gather(logits, 1, cb).float().cpu()

        results["BC"] = metrics_from_scores_stream(
            bc_score_fn, obs_bc, cand_bc, eval_batch=args.eval_batch
        )

    # --------------------------------------------------------
    # IQL (Q)
    # --------------------------------------------------------
    if args.iql_ckpt:
        print("Evaluating IQL...")
        ckpt = torch.load(args.iql_ckpt, map_location="cpu")
        state_dim = int(ckpt["state_dim"])
        num_items = int(ckpt["num_items"])

        obs_iql, cand_iql = filter_oob(obs, candidates, num_items=num_items, tag="IQL")

        class QNetwork(nn.Module):
            def __init__(self, num_items, state_dim, emb_dim=64, hidden1=256, hidden2=128):
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

        Q = QNetwork(num_items=num_items, state_dim=state_dim).to(DEVICE)
        Q.load_state_dict(ckpt["Q"])
        Q.eval()

        @torch.no_grad()
        def q_score_fn(ob, cb):
            # ob: [B, D], cb: [B, C]
            ob = ob.to(DEVICE, non_blocking=True)
            cb = cb.to(DEVICE, non_blocking=True)

            B, C = cb.shape

            # Compute Q(s,a) in chunks over candidates to control memory
            # Flatten over candidate axis in small blocks
            out = torch.empty((B, C), device="cpu", dtype=torch.float32)

            # Candidate-chunking (101 is small, but keeps peak mem down)
            cand_chunk = 25
            for j in range(0, C, cand_chunk):
                cb_j = cb[:, j:j + cand_chunk]  # [B, c]
                c = cb_j.shape[1]

                ob_rep = ob.unsqueeze(1).expand(-1, c, -1).reshape(B * c, -1)
                a_flat = cb_j.reshape(B * c)

                q_flat = Q(ob_rep, a_flat).reshape(B, c)  # on GPU
                out[:, j:j + c] = q_flat.float().cpu()

            return out

        results["IQL"] = metrics_from_scores_stream(
            q_score_fn, obs_iql, cand_iql, eval_batch=args.eval_batch
        )

    # --------------------------------------------------------
    # AWR policy
    # --------------------------------------------------------
    if args.awr_ckpt:
        print("Evaluating AWR...")
        ckpt = torch.load(args.awr_ckpt, map_location="cpu")
        state_dim = int(ckpt["state_dim"])
        num_items = int(ckpt["num_items"])

        obs_awr, cand_awr = filter_oob(obs, candidates, num_items=num_items, tag="AWR")

        class PolicyNet(nn.Module):
            def __init__(self, state_dim, num_items, hidden1=256, hidden2=128):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(state_dim, hidden1),
                    nn.ReLU(),
                    nn.Linear(hidden1, hidden2),
                    nn.ReLU(),
                    nn.Linear(hidden2, num_items),
                )

            def forward(self, s):
                return self.net(s)

        pi = PolicyNet(state_dim=state_dim, num_items=num_items).to(DEVICE)
        pi.load_state_dict(ckpt["policy"])
        pi.eval()

        @torch.no_grad()
        def awr_score_fn(ob, cb):
            ob = ob.to(DEVICE, non_blocking=True)
            cb = cb.to(DEVICE, non_blocking=True)
            logits = pi(ob)  # [B, num_items]
            return torch.gather(logits, 1, cb).float().cpu()

        results["AWR"] = metrics_from_scores_stream(
            awr_score_fn, obs_awr, cand_awr, eval_batch=args.eval_batch
        )

    # --------------------------------------------------------
    # PRINT + SAVE
    # --------------------------------------------------------
    print("\n====================================")
    print(f"RESULTS ({args.split})")
    print("====================================")
    for model_name, metrics in results.items():
        print(model_name)
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
        print()

    out_path = dataset_dir / f"metrics_{args.split}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print("Saved metrics to:", out_path)


if __name__ == "__main__":
    main()