# src/dataset_pipeline/gru_train.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .gru_dataset import NextItemDataset
from .gru_model import GRUEncoder, NextItemHead


def train_gru(events, num_items, cfg, device, output_dir):

    train_df = events[events["split"] == "train"]

    dataset = NextItemDataset(
        train_df,
        cfg.max_seq_len,
        cfg.pad_item_id
    )

    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True
    )

    encoder = GRUEncoder(
        num_items,
        cfg.emb_dim,
        cfg.hid_dim,
        cfg.pad_item_id
    ).to(device)

    head = NextItemHead(
        cfg.hid_dim,
        num_items
    ).to(device)

    opt = torch.optim.Adam(
        list(encoder.parameters()) + list(head.parameters()),
        lr=cfg.lr
    )

    ce = nn.CrossEntropyLoss()

    print("Training GRU...")

    for ep in range(cfg.epochs):

        encoder.train()
        head.train()

        total = 0
        n = 0

        for xb, yb in loader:

            xb = xb.to(device)
            yb = yb.to(device)

            opt.zero_grad()

            _, h = encoder(xb)

            logits = head(h[0])

            loss = ce(logits, yb)

            loss.backward()

            opt.step()

            total += loss.item() * xb.size(0)

            n += xb.size(0)

        print(f"Epoch {ep+1} loss {total/n:.4f}")

    torch.save(
        {
            "encoder_state_dict": encoder.state_dict(),
            "emb_dim": cfg.emb_dim,
            "hid_dim": cfg.hid_dim,
            "num_items": num_items
        },
        output_dir / "gru_encoder.pt"
    )

    return encoder