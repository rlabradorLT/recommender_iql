import os
import json
import argparse
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import yaml


# ============================================================
# CONFIG SYSTEM
# ============================================================

@dataclass
class DatasetConfig:
    experiment_name: str
    raw_path: str
    output_root: str

    pad_item_id: int
    max_seq_len: int

    emb_dim: int
    hid_dim: int

    batch_size: int
    epochs: int
    lr: float

    reward_type: str        # centered | scaled
    val_ratio: float
    test_ratio: float


def load_config(path: str):
    with open(path) as f:
        data = yaml.safe_load(f)
    return DatasetConfig(**data)


def create_output_dir(cfg: DatasetConfig):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    cfg_str = json.dumps(asdict(cfg), sort_keys=True)
    cfg_hash = str(abs(hash(cfg_str)))[:6]

    run_name = f"{cfg.experiment_name}_{cfg_hash}_{timestamp}"
    out_dir = Path(cfg.output_root) / run_name
    out_dir.mkdir(parents=True, exist_ok=False)

    with open(out_dir / "config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    return out_dir


# ============================================================
# ARGPARSE
# ============================================================

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True)
args = parser.parse_args()

cfg = load_config(args.config)
OUTPUT_DIR = create_output_dir(cfg)

device = "cuda" if torch.cuda.is_available() else "cpu"

PAD_ITEM_ID = cfg.pad_item_id
MAX_SEQ_LEN = cfg.max_seq_len
EMB_DIM = cfg.emb_dim
HID_DIM = cfg.hid_dim
BATCH_SIZE = cfg.batch_size
EPOCHS = cfg.epochs
LR = cfg.lr

print("Using config:", cfg)
print("Output dir:", OUTPUT_DIR)


# ============================================================
# 1️⃣ LOAD RAW
# ============================================================

def load_json(path):
    data = []
    with open(path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return pd.DataFrame(data)


print("Loading raw...")
df = load_json(cfg.raw_path)

# ============================================================
# 2️⃣ CLEAN
# ============================================================

df = df.dropna(subset=["reviewerID", "asin", "unixReviewTime", "overall"]).copy()
df["overall"] = df["overall"].astype(float).clip(1, 5)

# ============================================================
# 3️⃣ MAP IDS
# ============================================================

user2id = {u: i for i, u in enumerate(df["reviewerID"].unique())}
asin2id = {a: i + 1 for i, a in enumerate(df["asin"].unique())}  # 0 reserved

events = df[["reviewerID", "asin", "unixReviewTime", "overall"]].copy()
events["user_id"] = events["reviewerID"].map(user2id).astype(np.int32)
events["item_id"] = events["asin"].map(asin2id).astype(np.int32)

# ============================================================
# 4️⃣ REWARD CONFIGURABLE
# ============================================================

if cfg.reward_type == "centered":
    events["reward"] = ((events["overall"] - 1.0) / 4.0).clip(0.0, 1.0)
    user_mean = events.groupby("user_id")["reward"].transform("mean")
    events["reward"] = events["reward"] - user_mean

elif cfg.reward_type == "scaled":
    events["reward"] = ((events["overall"] - 1.0) / 4.0).clip(0.0, 1.0)

else:
    raise ValueError("Unknown reward_type")

print("Reward stats:")
print(" mean:", events["reward"].mean())
print(" std :", events["reward"].std())
print(" min :", events["reward"].min())
print(" max :", events["reward"].max())

events = events.sort_values(["user_id", "unixReviewTime"], kind="mergesort").reset_index(drop=True)

events["session_id"] = events["user_id"]
events["rank_in_user"] = events.groupby("user_id").cumcount()
events["user_len"] = events.groupby("user_id")["item_id"].transform("size")

# ============================================================
# 5️⃣ SPLIT CONFIGURABLE
# ============================================================

test_cut = (events["user_len"] * cfg.test_ratio).astype(int)
val_cut = (events["user_len"] * cfg.val_ratio).astype(int)

events["split"] = "train"
events.loc[events["rank_in_user"] >= test_cut, "split"] = "test"
events.loc[
    (events["rank_in_user"] >= val_cut) &
    (events["rank_in_user"] < test_cut),
    "split"
] = "val"

print("Users:", events["user_id"].nunique())
print("Items:", events["item_id"].nunique())

NUM_ITEMS = events["item_id"].nunique() + 1


# ============================================================
# 6️⃣ GRU PRETRAIN DATASET
# ============================================================

class NextItemDataset(Dataset):
    def __init__(self, events_df, max_len):
        self.max_len = max_len
        self.groups = []

        for uid, g in events_df.groupby("session_id", sort=False):
            items = g["item_id"].to_numpy(dtype=np.int64)
            if len(items) >= 2:
                self.groups.append(items)

        self.index = []
        for gi, items in enumerate(self.groups):
            for t in range(1, len(items)):
                self.index.append((gi, t))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        gi, t = self.index[idx]
        items = self.groups[gi]
        hist = items[:t][-self.max_len:]
        target = items[t]

        x = np.full((self.max_len,), PAD_ITEM_ID, dtype=np.int64)
        x[-len(hist):] = hist
        return torch.from_numpy(x), torch.tensor(target, dtype=torch.long)


train_df = events[events["split"] == "train"]
train_ds = NextItemDataset(train_df, MAX_SEQ_LEN)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

# ============================================================
# 7️⃣ MODEL
# ============================================================

class GRUEncoder(nn.Module):
    def __init__(self, num_items, emb_dim, hid_dim):
        super().__init__()
        self.emb = nn.Embedding(num_items, emb_dim, padding_idx=PAD_ITEM_ID)
        self.gru = nn.GRU(emb_dim, hid_dim, batch_first=True)

    def forward(self, x, h0=None):
        e = self.emb(x)
        out, h = self.gru(e, h0)
        return out, h


class NextItemHead(nn.Module):
    def __init__(self, hid_dim, num_items):
        super().__init__()
        self.fc = nn.Linear(hid_dim, num_items)

    def forward(self, h):
        return self.fc(h)


encoder = GRUEncoder(NUM_ITEMS, EMB_DIM, HID_DIM).to(device)
head = NextItemHead(HID_DIM, NUM_ITEMS).to(device)

opt = torch.optim.Adam(list(encoder.parameters()) + list(head.parameters()), lr=LR)
ce = nn.CrossEntropyLoss()

print("Training GRU...")
for ep in range(EPOCHS):
    encoder.train(); head.train()
    total = 0
    n = 0

    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)

        opt.zero_grad()
        _, h = encoder(xb)
        logits = head(h[0])
        loss = ce(logits, yb)
        loss.backward()
        opt.step()

        total += loss.item() * xb.size(0)
        n += xb.size(0)

    print(f"Epoch {ep+1} loss {total/n:.4f}")

torch.save({
    "encoder_state_dict": encoder.state_dict(),
    "emb_dim": EMB_DIM,
    "hid_dim": HID_DIM,
    "num_items": NUM_ITEMS
}, OUTPUT_DIR / "gru_encoder.pt")


# ============================================================
# 8️⃣ BUILD RL DATASET
# ============================================================

@torch.no_grad()
def build_rl_split(split_name):
    encoder.eval()

    obs_list, next_obs_list = [], []
    act_list, rew_list, done_list = [], [], []

    split_df = events[events["split"] == split_name]

    for uid, g in split_df.groupby("session_id", sort=False):
        items = g["item_id"].to_numpy(dtype=np.int64)
        rewards = g["reward"].to_numpy(dtype=np.float32)

        if len(items) < 2:
            continue

        h = torch.zeros((1, 1, HID_DIM), device=device)

        x0 = torch.tensor([[items[0]]], device=device)
        _, h = encoder(x0, h)

        for t in range(1, len(items)):
            s = h[0, 0].cpu().numpy().astype(np.float32)
            a = items[t]
            r = rewards[t]

            xt = torch.tensor([[a]], device=device)
            _, h2 = encoder(xt, h)
            s2 = h2[0, 0].cpu().numpy().astype(np.float32)

            done = 1.0 if (t == len(items) - 1) else 0.0

            obs_list.append(s)
            act_list.append(a)
            rew_list.append(r)
            next_obs_list.append(s2)
            done_list.append(done)

            h = h2

    return {
        "observations": np.stack(obs_list),
        "actions": np.array(act_list, dtype=np.int32),
        "rewards": np.array(rew_list, dtype=np.float32),
        "next_observations": np.stack(next_obs_list),
        "terminals": np.array(done_list, dtype=np.float32),
        "num_items": np.int64(NUM_ITEMS),
    }


print("Building RL splits...")
rl_train = build_rl_split("train")
rl_val = build_rl_split("val")
rl_test = build_rl_split("test")


# ============================================================
# 9️⃣ NORMALIZE (train only)
# ============================================================

mean = rl_train["observations"].mean(axis=0, keepdims=True)
std = rl_train["observations"].std(axis=0, keepdims=True) + 1e-6

def normalize_split(split):
    split["observations"] = (split["observations"] - mean) / std
    split["next_observations"] = (split["next_observations"] - mean) / std
    return split

rl_train = normalize_split(rl_train)
rl_val = normalize_split(rl_val)
rl_test = normalize_split(rl_test)

np.savez(OUTPUT_DIR / "normalization_stats.npz", mean=mean, std=std)


# ============================================================
# 🔟 SAVE
# ============================================================

def save_npz(name, data):
    np.savez_compressed(
        OUTPUT_DIR / name,
        observations=data["observations"],
        actions=data["actions"],
        rewards=data["rewards"],
        next_observations=data["next_observations"],
        terminals=data["terminals"],
        num_items=data["num_items"],
    )

save_npz("rl_train.npz", rl_train)
save_npz("rl_val.npz", rl_val)
save_npz("rl_test.npz", rl_test)

print("DONE. Files saved in:", OUTPUT_DIR)