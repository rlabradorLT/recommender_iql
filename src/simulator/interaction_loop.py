import numpy as np

from simulator.metrics import EpisodeMetrics, SimulationMetrics


# ============================================================
# episodio individual
# ============================================================

def run_episode(
    user_model,
    agent,
    warmup_items,
    horizon=10,
    forbid_repeated=True,
):
    """
    Ejecuta un episodio agente-usuario.

    Parámetros
    ----------
    user_model : GRUUserModel
    agent : BaseAgent
    warmup_items : list[int]
        historial real inicial
    horizon : int
        pasos simulados
    forbid_repeated : bool
        evita recomendar items ya vistos
    """

    episode = EpisodeMetrics()

    # ---------------------------------------------------------
    # inicializar usuario
    # ---------------------------------------------------------

    user_model.reset()
    user_model.warmup(warmup_items)

    consumed_items = set(warmup_items)

    # ---------------------------------------------------------
    # interacción
    # ---------------------------------------------------------

    for _ in range(horizon):

        state = user_model.get_state()

        # -----------------------------------------------------
        # recomendación agente
        # -----------------------------------------------------

        if forbid_repeated:
            action = agent.recommend(
                state,
                forbidden_items=consumed_items
            )
        else:
            action = agent.recommend(state)

        # -----------------------------------------------------
        # respuesta usuario
        # -----------------------------------------------------

        accepted, user_prob = user_model.evaluate_recommendation(action)

        reward = 1.0 if accepted else 0.0

        # -----------------------------------------------------
        # determinar item consumido
        # -----------------------------------------------------

        if accepted:

            next_item = action

        else:

            next_item = user_model.sample_next_item(
                exclude=list(consumed_items)
            )

        # -----------------------------------------------------
        # actualizar usuario
        # -----------------------------------------------------

        user_model.step(next_item)

        consumed_items.add(next_item)

        # -----------------------------------------------------
        # registrar métricas
        # -----------------------------------------------------

        episode.log_step(
            recommended_item=action,
            accepted=accepted,
            reward=reward,
            user_prob=user_prob,
        )

    return episode


# ============================================================
# simulación completa
# ============================================================

def run_simulation(
    user_model,
    agent,
    sessions,
    warmup_length=5,
    horizon=10,
):
    """
    Ejecuta simulación sobre muchas sesiones.

    Parámetros
    ----------
    sessions : list[list[int]]
        secuencias de items por sesión
    """

    sim_metrics = SimulationMetrics()

    valid_sessions = 0

    for seq in sessions:

        if len(seq) <= warmup_length:
            continue

        warmup = seq[:warmup_length]

        episode = run_episode(
            user_model=user_model,
            agent=agent,
            warmup_items=warmup,
            horizon=horizon,
        )

        sim_metrics.add_episode(episode)

        valid_sessions += 1

    return sim_metrics.compute()