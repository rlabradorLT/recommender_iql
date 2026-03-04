import numpy as np


def save_npz(output_dir, name, data):

    np.savez_compressed(
        output_dir / name,
        observations=data["observations"],
        actions=data["actions"],
        rewards=data["rewards"],
        next_observations=data["next_observations"],
        terminals=data["terminals"],
        num_items=data["num_items"],
    )


def save_dataset(output_dir, rl_train, rl_val, rl_test, events, mean, std):

    save_npz(output_dir, "rl_train.npz", rl_train)
    save_npz(output_dir, "rl_val.npz", rl_val)
    save_npz(output_dir, "rl_test.npz", rl_test)

    np.savez(
        output_dir / "normalization_stats.npz",
        mean=mean,
        std=std
    )

    events.to_parquet(output_dir / "events_with_split.parquet")