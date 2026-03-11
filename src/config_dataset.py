import yaml
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path

@dataclass
class DatasetConfig:
    experiment_name: str
    seed: int

    dataset: str
    raw_path: str
    output_root: str

    pad_item_id: int
    max_seq_len: int

    emb_dim: int
    hid_dim: int

    batch_size: int
    epochs: int
    lr: float
    save_head: bool

    reward_type: str
    val_ratio: float
    test_ratio: float

def load_config(path: str):
    with open(path) as f:
        data = yaml.safe_load(f)
    return DatasetConfig(**data)

def create_output_dir(cfg: DatasetConfig):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = f"{cfg.experiment_name}_{timestamp}"
    out_dir = Path(cfg.output_root) / run_name
    out_dir.mkdir(parents=True, exist_ok=False)

    with open(out_dir / "config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    return out_dir