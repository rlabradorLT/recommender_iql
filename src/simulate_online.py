import yaml
import random
import numpy as np
import torch
import pandas as pd

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


def load_sessions(parquet_path: str):
    df = pd.read_parquet(parquet_path)

    if "item_sequence" not in df.columns:
        raise ValueError("Dataset must contain column 'item_sequence'")

    sessions = df["item_sequence"].tolist()
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

    seeds = config["simulation"]["seeds"]

    results = []

    for seed in seeds:

        set_global_seed(seed)

        agent = load_agent(
            agent_type=agent_cfg["type"],
            checkpoint_path=agent_cfg["checkpoint"],
            stats_path=agent_cfg["normalization_stats"],
            score_transform=config["simulation"]["score_transform"],
        )

        user_model = GRUUserModel(
            checkpoint_path=config["user_model"]["checkpoint"],
            temperature=config["user_model"]["temperature"],
            acceptance_scale=config["user_model"]["acceptance_scale"],
            acceptance_min=config["user_model"]["acceptance_min"],
            acceptance_max=config["user_model"]["acceptance_max"],
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

    if agent.num_actions != user_model.num_items:
        raise RuntimeError(
            f"Agent action space ({agent.num_actions}) "
            f"!= user model items ({user_model.num_items})"
        )

    max_allowed = max(allowed_items)

    if max_allowed >= agent.num_actions:
        raise RuntimeError(
            f"allowed_items contains id {max_allowed} "
            f"outside agent action space ({agent.num_actions})"
        )


# ---------------------------------------------------------
# main
# ---------------------------------------------------------

def main():

    with open("simulate.yaml") as f:
        config = yaml.safe_load(f)

    sessions = load_sessions(config["dataset"]["sessions_path"])
    allowed_items = compute_allowed_items(sessions)

    print("===================================")
    print("ONLINE SIMULATION")
    print("===================================")

    print(f"Sessions: {len(sessions)}")
    print(f"Catalog size: {len(allowed_items)}")
    print()

    results = {}

    for agent_name, agent_cfg in config["agents"].items():

        print(f"Running agent: {agent_name}")

        agent = load_agent(
            agent_type=agent_cfg["type"],
            checkpoint_path=agent_cfg["checkpoint"],
            stats_path=agent_cfg["normalization_stats"],
        )

        user_model = GRUUserModel(
            checkpoint_path=config["user_model"]["checkpoint"]
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


if __name__ == "__main__":
    main()