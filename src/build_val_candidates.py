import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
import yaml


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_candidates(rl_npz_path, num_negatives):
    d = np.load(rl_npz_path)
    actions = d["actions"].reshape(-1).astype(np.int64)

    num_items = int(actions.max()) + 1
    N = len(actions)

    candidates = np.empty((N, num_negatives + 1), dtype=np.int32)

    for i in range(N):
        pos = actions[i]
        candidates[i, 0] = pos

        negs = np.random.randint(0, num_items, size=num_negatives)

        mask = (negs == pos)
        while mask.any():
            negs[mask] = np.random.randint(0, num_items, size=mask.sum())
            mask = (negs == pos)

        candidates[i, 1:] = negs

    return candidates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)

    dataset_dir = Path(cfg["dataset"]["dataset_dir"])
    num_negatives = cfg["candidates"]["num_negatives"]
    splits = cfg["candidates"]["splits"]

    for split in splits:
        print(f"Building {split} candidates from RL dataset...")
        cands = build_candidates(dataset_dir / f"rl_{split}.npz", num_negatives)
        np.save(dataset_dir / f"{split}_cands_rl.npy", cands)
        print(split, "shape:", cands.shape)


if __name__ == "__main__":
    main()