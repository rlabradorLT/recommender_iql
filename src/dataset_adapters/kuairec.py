# src/dataset_adapters/kuairec.py

import pandas as pd
from .base_adapter import DatasetAdapter


class KuaiRecAdapter(DatasetAdapter):

    def load_events(self, raw_path):

        df = pd.read_csv(raw_path)

        df = df.sort_values(["user_id","timestamp"])

        user2id = {u:i for i,u in enumerate(df.user_id.unique())}
        item2id = {v:i+1 for i,v in enumerate(df.video_id.unique())}

        events = pd.DataFrame({
            "user_id": df["user_id"].map(user2id),
            "item_id": df["video_id"].map(item2id),
            "timestamp": df["timestamp"],
            "reward": df["watch_ratio"]
        })

        return events