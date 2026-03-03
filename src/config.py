# src/config.py
import yaml
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import time

@dataclass
class BaseConfig:
    # general
    experiment_name: str = "default"
    seed: int = 7

    # paths
    data_path: str = "../data/processed/dataset_rl_by_split.npz"
    val_candidates_path: str = "../data/processed/val_candidates.npy"
    test_candidates_path: str = "../data/processed/test_candidates.npy"
    output_root: str = "../runs"

    # training
    batch_size: int = 256
    steps: int = 30000
    lr: float = 3e-4

def load_config(path: str) -> BaseConfig:
    with open(path, "r") as f:
        cfg_dict = yaml.safe_load(f)
    return BaseConfig(**cfg_dict)

def create_run_dir(cfg: BaseConfig):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = f"{cfg.experiment_name}_{timestamp}"
    run_dir = Path(cfg.output_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=False)

    # guardar config usada
    with open(run_dir / "config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    return run_dir