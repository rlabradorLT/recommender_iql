# src/dataset_adapters/kuairec.py

import pandas as pd
import numpy as np

from .base_adapter import DatasetAdapter


class KuaiRecAdapter(DatasetAdapter):

    def load_events(self, cfg):

        # ---------------------------------------------
        # LOAD
        # ---------------------------------------------

        df = pd.read_csv(cfg.raw_path)

        # ---------------------------------------------
        # CLEAN
        # ---------------------------------------------

        df = df.dropna(
            subset=["user_id", "video_id", "timestamp", "watch_ratio"]
        ).copy()

        df["watch_ratio"] = df["watch_ratio"].clip(0.0, 1.0)

        # ---------------------------------------------
        # MAP IDS
        # ---------------------------------------------

        user2id = {u: i for i, u in enumerate(df["user_id"].unique())}
        item2id = {v: i + 1 for i, v in enumerate(df["video_id"].unique())}

        events = pd.DataFrame()

        events["user_id"] = df["user_id"].map(user2id).astype(np.int32)
        events["item_id"] = df["video_id"].map(item2id).astype(np.int32)

        events["timestamp"] = df["timestamp"].astype(np.int64)

        # ---------------------------------------------
        # REWARD
        # ---------------------------------------------

        if cfg.reward_type == "scaled":

            events["reward"] = df["watch_ratio"].astype(np.float32)

        elif cfg.reward_type == "centered":

            events["reward"] = df["watch_ratio"].astype(np.float32)

            user_mean = events.groupby("user_id")["reward"].transform("mean")

            events["reward"] = events["reward"] - user_mean

        elif cfg.reward_type == "custom":
            events["reward"] = (
                df["watch_ratio"].astype(np.float32)
                + 0.5 * df["like"].astype(np.float32)
            ).clip(0.0, 1.5)
            
        else:
            raise ValueError("Unknown reward_type")

        # ---------------------------------------------

        return events[["user_id", "item_id", "timestamp", "reward"]]