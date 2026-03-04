# src/dataset_pipeline/splitter.py

from asyncio import events
import numpy as np


def add_splits(events, cfg):

    # --------------------------------------------
    # ordenar eventos
    # --------------------------------------------

    events = events.sort_values(
        ["user_id", "timestamp"],
        kind="mergesort"
    ).reset_index(drop=True)

    # --------------------------------------------
    # columnas auxiliares
    # --------------------------------------------

    events["session_id"] = events["user_id"]

    events["rank_in_user"] = (
        events.groupby("user_id").cumcount()
    )

    events["user_len"] = (
        events.groupby("user_id")["item_id"].transform("size")
    )

    # --------------------------------------------
    # split
    # --------------------------------------------

    test_cut = (events["user_len"] * (1 - cfg.test_ratio)).astype(int)
    val_cut = (events["user_len"] * (1 - cfg.test_ratio - cfg.val_ratio)).astype(int)

    events["split"] = "train"

    events.loc[
        events["rank_in_user"] >= test_cut,
        "split"
    ] = "test"

    events.loc[
        (events["rank_in_user"] >= val_cut) &
        (events["rank_in_user"] < test_cut),
        "split"
    ] = "val"

    print("Users:", events["user_id"].nunique())
    print("Items:", events["item_id"].nunique())
    print("\nSplit ratios (events):")
    print(events["split"].value_counts(normalize=True))

    print("\nSplit counts:")
    print(events["split"].value_counts())

    num_items = events["item_id"].nunique() + 1

    return events, num_items