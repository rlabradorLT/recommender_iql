import yaml
import random
import numpy as np
import torch
import pandas as pd
import argparse
import json
from pathlib import Path

from simulator.agent_loader import load_agent
from simulator.user_model import GRUUserModel
from simulator.interaction_loop import run_simulation


# ---------------------------------------------------------
# utilidades
# ---------------------------------------------------------

def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_sessions(events_path: str, split: str):
    df = pd.read_parquet(events_path)

    required_cols = {"session_id", "item_id", "split"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {sorted(missing)}"
        )

    df = df[df["split"] == split].copy()

    if len(df) == 0:
        raise ValueError(f"No events found for split={split!r}")

    order_cols = [c for c in ["event_idx", "timestamp", "ts"] if c in df.columns]

    if order_cols:
        df = df.sort_values(["session_id"] + order_cols)
    else:
        df = df.sort_values(["session_id"])

    sessions = (
        df.groupby("session_id")["item_id"]
        .apply(lambda x: [int(v) for v in x.tolist()])
        .tolist()
    )

    return sessions


def compute_allowed_items(sessions):

    items = set()

    for seq in sessions:
        for item in seq:
            items.add(int(item))

    return items


def aggregate_results(results):

    keys = results[0].keys()
    out = {}

    for k in keys:

        vals = [r[k] for r in results]

        mean = float(np.mean(vals))
        std = float(np.std(vals))
        ci95 = 1.96 * std / np.sqrt(len(vals))

        out[k] = {
            "mean": mean,
            "std": std,
            "ci95": float(ci95),
        }

    return out


# ---------------------------------------------------------
# experimento
# ---------------------------------------------------------

def run_agent_experiment(config, agent_name, agent_cfg, sessions, allowed_items):

    sim_cfg = config["simulation"]

    if "seeds" in sim_cfg:
        seeds = sim_cfg["seeds"]
    elif "seed" in sim_cfg:
        seeds = [sim_cfg["seed"]]
    else:
        seeds = [42]

    results = []

    for seed in seeds:

        set_global_seed(seed)

        agent = load_agent(
            agent_type=agent_cfg["type"],
            checkpoint_path=agent_cfg["checkpoint"],
            stats_path=config["dataset"]["normalization_stats"],
        )

        user_model = GRUUserModel(
            checkpoint_path=config["simulator"]["gru_checkpoint"],
            temperature=config["simulator"]["temperature"],
        )

        result = run_simulation(
            user_model=user_model,
            agent=agent,
            sessions=sessions,
            allowed_items=allowed_items,
            warmup_length=config["simulation"]["warmup_length"],
            horizon=config["simulation"]["horizon"],
            forbid_repeated=config["simulation"]["forbid_repeated"],
            acceptance_mode=config["simulation"]["acceptance_mode"],
            num_candidates=config["simulation"]["num_candidates"],
        )
        results.append(result)
    
    return aggregate_results(results)


# ---------------------------------------------------------
# validaciones
# ---------------------------------------------------------

def validate_environment(agent, user_model, allowed_items):

    if user_model.num_items > agent.num_actions:
        raise RuntimeError(
            f"user model items ({user_model.num_items}) "
            f"> agent action space ({agent.num_actions})"
        )

    max_allowed = max(allowed_items)

    if max_allowed >= user_model.num_items:
        raise RuntimeError(
            f"allowed_items contains id {max_allowed} "
            f"outside user model space ({user_model.num_items})"
        )

# ---------------------------------------------------------
# main
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        required=True
    )

    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    sessions = load_sessions(
        events_path=config["dataset"]["events_path"],
        split=config["simulation"]["split"],
    )
    allowed_items = compute_allowed_items(sessions)

    print("===================================")
    print("ONLINE SIMULATION")
    print("===================================")

    print(f"Sessions: {len(sessions)}")
    print(f"Catalog size: {len(allowed_items)}")
    print()

    results = {}

    for agent_cfg in config["agents"]:

        agent_name = agent_cfg["name"]

        print(f"Running agent: {agent_name}")

        agent = load_agent(
            agent_type=agent_cfg["type"],
            checkpoint_path=agent_cfg["checkpoint"],
            stats_path=config["dataset"]["normalization_stats"],
        )

        user_model = GRUUserModel(
            checkpoint_path=config["simulator"]["gru_checkpoint"],
            temperature=config["simulator"]["temperature"],
        )

        validate_environment(agent, user_model, allowed_items)

        agent_result = run_agent_experiment(
            config,
            agent_name,
            agent_cfg,
            sessions,
            allowed_items,
        )

        results[agent_name] = agent_result

    print()
    print("===================================")
    print("RESULTS")
    print("===================================")

    for agent, metrics in results.items():

        print()
        print(agent)

        for k, v in metrics.items():

            print(
                f"{k:30s} "
                f"{v['mean']:.4f} ± {v['ci95']:.4f} "
                f"(std={v['std']:.4f})"
            )

    # -------- guardar JSON --------

    output_path = Path(config["output"]["results_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print()
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()