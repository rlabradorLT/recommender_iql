import numpy as np


# ---------------------------------------------------------
# utilidades
# ---------------------------------------------------------

def intra_episode_repetition(items):
    """
    Mide repetición dentro de una secuencia.

    0.0 = todos los items distintos
    1.0 = todos iguales
    """
    if len(items) <= 1:
        return 0.0

    unique = len(set(items))
    return 1.0 - unique / len(items)


# ---------------------------------------------------------
# métricas por episodio
# ---------------------------------------------------------

class EpisodeMetrics:

    def __init__(self):

        self.recommended_items = []
        self.accepted = []
        self.rewards = []
        self.user_probs = []
        self.accept_probs = []

        self.catalog_size = None

    def log_step(
        self,
        recommended_item,
        accepted,
        reward,
        user_prob,
        accept_probability,
        catalog_size,
    ):

        self.recommended_items.append(int(recommended_item))
        self.accepted.append(bool(accepted))
        self.rewards.append(float(reward))
        self.user_probs.append(float(user_prob))
        self.accept_probs.append(float(accept_probability))

        if self.catalog_size is None:
            self.catalog_size = int(catalog_size)

    def compute(self):

        if len(self.recommended_items) == 0:
            return None

        recommended_items = self.recommended_items

        return {
            "reward": float(np.sum(self.rewards)),
            "acceptance_rate": float(np.mean(self.accepted)),
            "avg_user_probability": float(np.mean(self.user_probs)),
            "avg_accept_probability": float(np.mean(self.accept_probs)),
            "session_length": int(len(self.recommended_items)),
            "unique_items": int(len(set(recommended_items))),
            "intra_episode_repetition": intra_episode_repetition(
                recommended_items
            ),
            "catalog_size": int(self.catalog_size),
        }


# ---------------------------------------------------------
# métricas de simulación
# ---------------------------------------------------------

class SimulationMetrics:

    def __init__(self):

        self.episodes = []

        self.all_recommended_items = []
        self.all_rewards = []
        self.all_accepts = []
        self.all_user_probs = []
        self.all_accept_probs = []

        self.catalog_size = None

    def add_episode(self, episode: EpisodeMetrics):

        result = episode.compute()

        if result is None:
            return

        self.episodes.append(result)

        self.all_recommended_items.extend(episode.recommended_items)
        self.all_rewards.extend(episode.rewards)
        self.all_accepts.extend(episode.accepted)
        self.all_user_probs.extend(episode.user_probs)
        self.all_accept_probs.extend(episode.accept_probs)

        if self.catalog_size is None:
            self.catalog_size = result["catalog_size"]

    def compute(self):

        if len(self.all_recommended_items) == 0:
            return {}

        unique_items = len(set(self.all_recommended_items))
        catalog_size = self.catalog_size

        if catalog_size is None or catalog_size == 0:
            catalog_coverage = 0.0
        else:
            catalog_coverage = unique_items / catalog_size

        episode_lengths = [
            ep["session_length"]
            for ep in self.episodes
        ]

        intra_repetitions = [
            ep["intra_episode_repetition"]
            for ep in self.episodes
        ]

        return {

            "cumulative_reward":
                float(np.sum(self.all_rewards)),

            "acceptance_rate":
                float(np.mean(self.all_accepts)),

            "avg_user_probability":
                float(np.mean(self.all_user_probs)),

            "avg_accept_probability":
                float(np.mean(self.all_accept_probs)),

            "avg_session_length":
                float(np.mean(episode_lengths)),

            "catalog_coverage":
                float(catalog_coverage),

            "unique_items_recommended":
                int(unique_items),

            "avg_intra_episode_repetition":
                float(np.mean(intra_repetitions)),
        }