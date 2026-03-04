# src/dataset_pipeline/rl_builder.py

import numpy as np
import torch


@torch.no_grad()
def build_rl_split(events, split_name, encoder, hid_dim, device, num_items):

    encoder.eval()

    obs_list, next_obs_list = [], []
    act_list, rew_list, done_list = [], [], []

    split_df = events[events["split"] == split_name]

    for uid, g in split_df.groupby("session_id", sort=False):

        items = g["item_id"].to_numpy(dtype=np.int64)
        rewards = g["reward"].to_numpy(dtype=np.float32)

        if len(items) < 2:
            continue

        h = torch.zeros((1, 1, hid_dim), device=device)

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
        "num_items": np.int64(num_items),
    }