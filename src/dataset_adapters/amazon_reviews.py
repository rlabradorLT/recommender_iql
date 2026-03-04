# src/dataset_adapters/amazon_reviews.py

import json
import pandas as pd
import numpy as np

from .base_adapter import DatasetAdapter


class AmazonReviewsAdapter(DatasetAdapter):

    def load_events(self, cfg):

        data = []

        with open(cfg.raw_path) as f:
            for line in f:
                data.append(json.loads(line))

        df = pd.DataFrame(data)

        # ------------------------------------------------
        # CLEAN
        # ------------------------------------------------

        df = df.dropna(
            subset=["reviewerID", "asin", "unixReviewTime", "overall"]
        ).copy()

        df["overall"] = df["overall"].astype(float).clip(1, 5)

        # ------------------------------------------------
        # MAP IDS
        # ------------------------------------------------

        user2id = {u: i for i, u in enumerate(df["reviewerID"].unique())}
        item2id = {a: i + 1 for i, a in enumerate(df["asin"].unique())}

        events = df[["reviewerID", "asin", "unixReviewTime", "overall"]].copy()

        events["user_id"] = events["reviewerID"].map(user2id).astype(np.int32)
        events["item_id"] = events["asin"].map(item2id).astype(np.int32)

        # ------------------------------------------------
        # REWARD
        # ------------------------------------------------

        if cfg.reward_type == "centered":

            events["reward"] = ((events["overall"] - 1.0) / 4.0).clip(0.0, 1.0)

            user_mean = events.groupby("user_id")["reward"].transform("mean")

            events["reward"] = events["reward"] - user_mean

        elif cfg.reward_type == "scaled":

            events["reward"] = ((events["overall"] - 1.0) / 4.0).clip(0.0, 1.0)

        else:
            raise ValueError("Unknown reward_type")

        # ------------------------------------------------

        events["timestamp"] = events["unixReviewTime"]

        return events[["user_id", "item_id", "timestamp", "reward"]]