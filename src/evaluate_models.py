import argparse
import json
from pathlib import Path

import yaml
import numpy as np
import torch
from tqdm import tqdm

from dqn.q_network import QNetwork
from dqn.policiy_network import PolicyNet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# METRICS ENGINE
# ============================================================

@torch.no_grad()
def compute_ranking_metrics(score_fn, obs, candidates, batch_size, ks):

    sums = {}

    for k in ks:
        sums[f"HR@{k}"] = 0.0
        sums[f"NDCG@{k}"] = 0.0

    sums["MRR"] = 0.0
    sums["MeanRank"] = 0.0

    total = 0

    for i in tqdm(range(0, obs.shape[0], batch_size), desc="Eval", leave=False):

        ob = obs[i:i+batch_size]
        cb = candidates[i:i+batch_size]

        scores = score_fn(ob, cb)

        pos_scores = scores[:, 0:1]

        better = (scores > pos_scores).sum(dim=1)

        rank = better + 1

        for k in ks:

            hit = (rank <= k).float()

            sums[f"HR@{k}"] += hit.sum().item()

            ndcg = hit / torch.log2(rank.float() + 1)

            sums[f"NDCG@{k}"] += ndcg.sum().item()

        sums["MRR"] += (1.0 / rank.float()).sum().item()
        sums["MeanRank"] += rank.sum().item()

        total += scores.shape[0]

    return {k: v / total for k, v in sums.items()}


# ============================================================
# DATA
# ============================================================

def load_split(dataset_dir, split):

    data = np.load(dataset_dir / f"rl_{split}.npz")

    obs = torch.tensor(data["observations"], dtype=torch.float32)

    candidates = torch.tensor(
        np.load(dataset_dir / f"{split}_cands_rl.npy"),
        dtype=torch.long
    )

    num_items = int(data["num_items"])

    return obs, candidates, num_items


# ============================================================
# MODEL BUILDERS
# ============================================================

def build_iql_scorer(ckpt_path, num_items, hidden1, hidden2):

    ckpt = torch.load(ckpt_path, map_location=DEVICE)

    state_dim = ckpt["state_dim"]

    Q1 = QNetwork(state_dim, num_items, hidden1, hidden2).to(DEVICE)
    Q2 = QNetwork(state_dim, num_items, hidden1, hidden2).to(DEVICE)

    Q1.load_state_dict(ckpt["Q1"])
    Q2.load_state_dict(ckpt["Q2"])

    Q1.eval()
    Q2.eval()

    @torch.no_grad()
    def score_fn(ob, cb):

        ob = ob.to(DEVICE)
        cb = cb.to(DEVICE)

        q1 = Q1(ob)
        q2 = Q2(ob)

        q = torch.minimum(q1, q2)

        return torch.gather(q, 1, cb).cpu()

    return score_fn


def build_policy_scorer(ckpt_path, num_items, hidden1, hidden2):

    ckpt = torch.load(ckpt_path, map_location=DEVICE)

    state_dim = ckpt["state_dim"]

    policy = PolicyNet(state_dim, num_items, hidden1, hidden2).to(DEVICE)

    policy.load_state_dict(ckpt["policy"])

    policy.eval()

    @torch.no_grad()
    def score_fn(ob, cb):

        ob = ob.to(DEVICE)
        cb = cb.to(DEVICE)

        logits = policy(ob)

        return torch.gather(logits, 1, cb).cpu()

    return score_fn


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))

    dataset_dir = Path(cfg["dataset_dir"])

    split = cfg.get("split", "val")
    batch_size = cfg.get("eval_batch", 2048)
    ks = cfg.get("ks", [5,10,20])

    hidden1 = cfg.get("hidden1", 512)
    hidden2 = cfg.get("hidden2", 512)

    obs, candidates, num_items = load_split(dataset_dir, split)

    results = {}

    # ============================================================
    # MODELS
    # ============================================================

    for name, model_cfg in cfg["models"].items():

        print(f"\nEvaluating {name}")

        mtype = model_cfg["type"]
        ckpt = model_cfg["ckpt"]

        if mtype == "critic":

            scorer = build_iql_scorer(
                ckpt,
                num_items,
                hidden1,
                hidden2
            )

        elif mtype == "policy":

            scorer = build_policy_scorer(
                ckpt,
                num_items,
                hidden1,
                hidden2
            )

        else:
            raise ValueError(f"Unknown model type {mtype}")

        results[name] = compute_ranking_metrics(
            scorer,
            obs,
            candidates,
            batch_size,
            ks
        )

    # ============================================================
    # PRINT
    # ============================================================

    print("\n==============================")
    print(f"RESULTS ({split})")
    print("==============================")

    for model, metrics in results.items():

        print(model)

        for k, v in metrics.items():
            print(f"  {k}: {v:.6f}")

        print()

    out_path = dataset_dir / f"metrics_{split}.json"

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print("Saved metrics:", out_path)


if __name__ == "__main__":
    main()