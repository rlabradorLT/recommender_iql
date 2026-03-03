import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm


NUM_NEGATIVES = 100


def build_candidates(rl_npz_path):
    d = np.load(rl_npz_path)
    actions = d["actions"].reshape(-1).astype(np.int64)

    num_items = int(actions.max()) + 1
    N = len(actions)

    candidates = np.empty((N, 101), dtype=np.int32)

    for i in range(N):
        pos = actions[i]
        candidates[i, 0] = pos

        negs = np.random.randint(0, num_items, size=100)

        # evitar que el positivo aparezca en negativos
        mask = (negs == pos)
        while mask.any():
            negs[mask] = np.random.randint(0, num_items, size=mask.sum())
            mask = (negs == pos)

        candidates[i, 1:] = negs

    return candidates

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)

    for split in ["val", "test"]:
        print(f"Building {split} candidates from RL dataset...")
        cands = build_candidates(dataset_dir / f"rl_{split}.npz")
        np.save(dataset_dir / f"{split}_cands_rl.npy", cands)
        print(split, "shape:", cands.shape)


if __name__ == "__main__":
    main()