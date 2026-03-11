# src/dataset_pipeline/normalization.py

import numpy as np


def normalize_rl_datasets(rl_train, rl_val, rl_test):

    mean = rl_train["observations"].mean(axis=0, keepdims=True)

    std = rl_train["observations"].std(axis=0, keepdims=True) + 1e-6

    def normalize_split(split):

        split["observations"] = (split["observations"] - mean) / std

        split["next_observations"] = (
            split["next_observations"] - mean
        ) / std

        return split

    rl_train = normalize_split(rl_train)
    rl_val = normalize_split(rl_val)
    rl_test = normalize_split(rl_test)

    return rl_train, rl_val, rl_test, mean, std

# ------------------------------------------------------------
# cargar estadísticas de normalización
# ------------------------------------------------------------

def load_normalization_stats(path):

    data = np.load(path)

    mean = data["mean"]
    std = data["std"]

    return mean, std


# ------------------------------------------------------------
# normalizar una observación individual
# ------------------------------------------------------------

def normalize_obs(obs, stats):

    mean, std = stats

    return (obs - mean) / (std + 1e-6)