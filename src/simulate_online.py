import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import torch

from simulator.user_model import GRUUserModel
from simulator.agent_loader import load_agent
from simulator.interaction_loop import run_simulation


# ============================================================
# utilidades
# ============================================================

def set_seed(seed):

    np.random.seed(seed)
    torch.manual_seed(seed)


def load_sessions(events_path, split="test"):
    """
    Carga sesiones desde events_with_split.parquet
    """

    df = pd.read_parquet(events_path)

    df = df[df["split"] == split]

    sessions = []

    for _, g in df.groupby("session_id"):

        seq = g.sort_values("timestamp")["item_id"].tolist()

        sessions.append(seq)

    return sessions


# ============================================================
# main
# ============================================================

def main(cfg):

    seed = cfg["simulation"]["seed"]
    set_seed(seed)

    events_path = cfg["dataset"]["events_path"]
    norm_stats = cfg["dataset"]["normalization_stats"]

    split = cfg["simulation"]["split"]
    warmup_length = cfg["simulation"]["warmup_length"]
    horizon = cfg["simulation"]["horizon"]

    print("\nLoading sessions...")

    sessions = load_sessions(events_path, split)

    print(f"Sessions loaded: {len(sessions)}")

    # --------------------------------------------------------
    # simulador usuario
    # --------------------------------------------------------

    print("\nLoading GRU user simulator...")

    user_model = GRUUserModel(
        checkpoint_path=cfg["simulator"]["gru_checkpoint"],
        temperature=cfg["simulator"]["temperature"],
    )

    # --------------------------------------------------------
    # ejecutar simulaciones
    # --------------------------------------------------------

    all_results = {}

    for agent_cfg in cfg["agents"]:

        name = agent_cfg["name"]
        agent_type = agent_cfg["type"]
        checkpoint = agent_cfg["checkpoint"]

        print("\n==============================")
        print(f"Simulating agent: {name}")
        print("==============================")

        # reset seed para comparabilidad
        set_seed(seed)

        agent = load_agent(
            agent_type=agent_type,
            checkpoint_path=checkpoint,
            stats_path=norm_stats,
        )

        results = run_simulation(
            user_model=user_model,
            agent=agent,
            sessions=sessions,
            warmup_length=warmup_length,
            horizon=horizon,
        )

        all_results[name] = results

        for k, v in results.items():
            print(f"{k:25s}: {v:.4f}")

    # --------------------------------------------------------
    # guardar resultados
    # --------------------------------------------------------

    output_path = cfg["output"]["results_path"]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print("\nResults saved to:", output_path)


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        required=True,
    )

    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    main(cfg)