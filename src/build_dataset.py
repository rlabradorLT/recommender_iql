import argparse
import torch

from config_dataset import load_config, create_output_dir

from dataset_adapters import load_adapter

from dataset_pipeline.splitter import add_splits
from dataset_pipeline.gru_train import train_gru
from dataset_pipeline.rl_builder import build_rl_split
from dataset_pipeline.normalization import normalize_rl_datasets
from dataset_pipeline.saver import save_dataset


# ============================================================
# ARGPARSE
# ============================================================

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True)

args = parser.parse_args()

cfg = load_config(args.config)

torch.manual_seed(cfg.seed)

OUTPUT_DIR = create_output_dir(cfg)

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Using config:", cfg)
print("Output dir:", OUTPUT_DIR)
print("Device:", device)


# ============================================================
# 1️⃣ LOAD DATASET (ADAPTER)
# ============================================================

print("Loading dataset adapter...")

adapter = load_adapter(cfg.dataset)

events = adapter.load_events(cfg)

print("Loaded events:", len(events))


# ============================================================
# 2️⃣ BUILD SPLITS
# ============================================================

print("Building splits...")

events, NUM_ITEMS = add_splits(events, cfg)


# ============================================================
# 3️⃣ TRAIN GRU ENCODER
# ============================================================

encoder = train_gru(
    events=events,
    num_items=NUM_ITEMS,
    cfg=cfg,
    device=device,
    output_dir=OUTPUT_DIR
)


# ============================================================
# 4️⃣ BUILD RL DATASET
# ============================================================

print("Building RL splits...")

rl_train = build_rl_split(
    events,
    "train",
    encoder,
    cfg.hid_dim,
    device,
    NUM_ITEMS
)

rl_val = build_rl_split(
    events,
    "val",
    encoder,
    cfg.hid_dim,
    device,
    NUM_ITEMS
)

rl_test = build_rl_split(
    events,
    "test",
    encoder,
    cfg.hid_dim,
    device,
    NUM_ITEMS
)


# ============================================================
# 5️⃣ NORMALIZE
# ============================================================

rl_train, rl_val, rl_test, mean, std = normalize_rl_datasets(
    rl_train,
    rl_val,
    rl_test
)


# ============================================================
# 6️⃣ SAVE DATASET
# ============================================================

save_dataset(
    OUTPUT_DIR,
    rl_train,
    rl_val,
    rl_test,
    events,
    mean,
    std
)

print("DONE. Files saved in:", OUTPUT_DIR)