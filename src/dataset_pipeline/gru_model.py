# src/dataset_pipeline/gru_model.py

import torch
import torch.nn as nn


class GRUEncoder(nn.Module):

    def __init__(self, num_items, emb_dim, hid_dim, pad_item_id):

        super().__init__()

        self.emb = nn.Embedding(
            num_items,
            emb_dim,
            padding_idx=pad_item_id
        )

        self.gru = nn.GRU(
            emb_dim,
            hid_dim,
            batch_first=True
        )


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