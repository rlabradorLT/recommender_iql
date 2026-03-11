import numpy as np


# ============================================================
# métricas por episodio
# ============================================================

class EpisodeMetrics:
    """
    Métricas de un solo episodio (usuario).
    """

    def __init__(self):

        self.steps = 0
        self.accepts = 0
        self.reward = 0.0

        self.user_probs = []

        self.recommended_items = []
        self.accepted_items = []

    # ---------------------------------------------------------
    # registrar paso
    # ---------------------------------------------------------

    def log_step(
        self,
        recommended_item,
        accepted,
        reward,
        user_prob
    ):

        self.steps += 1
        self.reward += reward

        self.recommended_items.append(recommended_item)
        self.user_probs.append(user_prob)

        if accepted:
            self.accepts += 1
            self.accepted_items.append(recommended_item)

    # ---------------------------------------------------------
    # resultados episodio
    # ---------------------------------------------------------

    def compute(self):

        results = {}

        if self.steps == 0:
            return results

        results["reward"] = float(self.reward)

        results["acceptance_rate"] = float(
            self.accepts / self.steps
        )

        results["avg_user_probability"] = float(
            np.mean(self.user_probs)
        )

        results["episode_length"] = self.steps

        results["unique_items"] = len(set(self.recommended_items))

        results["repetition_rate"] = repetition_rate(
            self.recommended_items
        )

        return results


# ============================================================
# métricas globales
# ============================================================

class SimulationMetrics:
    """
    Agrega métricas de todos los episodios.
    """

    def __init__(self):

        self.total_steps = 0
        self.total_reward = 0.0
        self.total_accepts = 0

        self.user_probs = []

        self.recommended_items = []

        self.session_lengths = []

    # ---------------------------------------------------------
    # añadir episodio
    # ---------------------------------------------------------

    def add_episode(self, episode: EpisodeMetrics):

        res = episode.compute()

        if not res:
            return

        self.total_steps += episode.steps
        self.total_reward += episode.reward
        self.total_accepts += episode.accepts

        self.user_probs.extend(episode.user_probs)
        self.recommended_items.extend(episode.recommended_items)

        self.session_lengths.append(episode.steps)

    # ---------------------------------------------------------
    # métricas finales
    # ---------------------------------------------------------

    def compute(self):

        results = {}

        if self.total_steps == 0:
            return results

        # ---------------------------------------------
        # reward acumulada
        # ---------------------------------------------

        results["cumulative_reward"] = float(self.total_reward)

        # ---------------------------------------------
        # acceptance rate
        # ---------------------------------------------

        results["acceptance_rate"] = float(
            self.total_accepts / self.total_steps
        )

        # ---------------------------------------------
        # probabilidad media usuario
        # ---------------------------------------------

        if self.user_probs:

            results["avg_user_probability"] = float(
                np.mean(self.user_probs)
            )

        # ---------------------------------------------
        # longitud media sesión
        # ---------------------------------------------

        if self.session_lengths:

            results["avg_session_length"] = float(
                np.mean(self.session_lengths)
            )

        # ---------------------------------------------
        # cobertura catálogo
        # ---------------------------------------------

        if self.recommended_items:

            unique_items = len(set(self.recommended_items))
            total = len(self.recommended_items)

            results["item_coverage"] = float(unique_items / total)

            results["unique_items_recommended"] = unique_items

        # ---------------------------------------------
        # repetición
        # ---------------------------------------------

        if self.recommended_items:

            results["repetition_rate"] = repetition_rate(
                self.recommended_items
            )

        return results


# ============================================================
# utilidades
# ============================================================

def repetition_rate(items):
    """
    Proporción de recomendaciones repetidas.
    """

    seen = set()
    repeats = 0

    for item in items:

        if item in seen:
            repeats += 1

        seen.add(item)

    return repeats / max(len(items), 1)