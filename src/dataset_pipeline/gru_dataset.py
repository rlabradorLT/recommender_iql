# src/dataset_pipeline/gru_dataset.py

import numpy as np
import torch
from torch.utils.data import Dataset


class NextItemDataset(Dataset):

    def __init__(self, events_df, max_len, pad_item_id):

        self.max_len = max_len
        self.pad_item_id = pad_item_id

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

        x = np.full((self.max_len,), self.pad_item_id, dtype=np.int64)

        x[-len(hist):] = hist

        return torch.from_numpy(x), torch.tensor(target, dtype=torch.long)