import numpy as np

from simulator.metrics import EpisodeMetrics, SimulationMetrics


def _build_forbidden_items(
    num_actions: int,
    allowed_items,
    consumed_items=None,
    forbid_repeated: bool = True,
):
    """
    Construye el conjunto de items prohibidos para TODOS los agentes
    bajo exactamente las mismas reglas.

    Reglas:
    - todo item fuera de allowed_items queda prohibido
    - opcionalmente, items ya consumidos quedan prohibidos
    """
    allowed_items = np.asarray(sorted(set(allowed_items)), dtype=np.int64)

    if allowed_items.size == 0:
        raise ValueError("allowed_items is empty.")

    valid_allowed = allowed_items[
        (allowed_items >= 0) & (allowed_items < num_actions)
    ]

    if valid_allowed.size == 0:
        raise ValueError(
            "No allowed_items are valid under the agent action space."
        )

    allowed_set = set(valid_allowed.tolist())

    forbidden = set(i for i in range(num_actions) if i not in allowed_set)

    if forbid_repeated and consumed_items:
        for item in consumed_items:
            if 0 <= item < num_actions:
                forbidden.add(int(item))

    return forbidden, valid_allowed


def sample_candidates_without_replacement(action, valid_items, num_candidates=100):
    """
    Construye un candidate set SIN reemplazo, incluyendo siempre la acción elegida.

    Esto deja disponible el protocolo por candidate set, pero corrige el sesgo
    de repetir negativos aleatorios.
    """
    valid_items = np.asarray(valid_items, dtype=np.int64)

    if valid_items.ndim != 1:
        raise ValueError("valid_items must be 1D.")

    if valid_items.size == 0:
        raise ValueError("valid_items is empty.")

    pool = valid_items[valid_items != action]

    max_total = min(num_candidates, valid_items.size)

    if max_total <= 1:
        return np.asarray([action], dtype=np.int64)

    num_negatives = max_total - 1

    if pool.size <= num_negatives:
        negatives = pool
    else:
        negatives = np.random.choice(
            pool,
            size=num_negatives,
            replace=False,
        )

    candidates = np.concatenate(
        [np.asarray([action], dtype=np.int64), negatives.astype(np.int64)]
    )

    return candidates


def run_episode(
    user_model,
    agent,
    warmup_items,
    allowed_items,
    horizon=10,
    forbid_repeated=True,
    acceptance_mode="direct",
    num_candidates=100,
):

    episode = EpisodeMetrics()

    user_model.reset()
    user_model.warmup(warmup_items)

    consumed_items = set(warmup_items)

    for _ in range(horizon):
        state = user_model.get_state()

        forbidden_items, valid_allowed = _build_forbidden_items(
            num_actions=agent.num_actions,
            allowed_items=allowed_items,
            consumed_items=consumed_items,
            forbid_repeated=forbid_repeated,
        )

        # Todos los agentes reciben exactamente la misma restricción
        action = agent.recommend(
            state,
            forbidden_items=forbidden_items,
        )

        if acceptance_mode == "direct":
            accepted, user_prob, accept_p = user_model.evaluate_recommendation(
                item_id=action,
                mode="direct",
            )

        elif acceptance_mode == "candidate_set":
            candidates = sample_candidates_without_replacement(
                action=action,
                valid_items=valid_allowed,
                num_candidates=num_candidates,
            )

            accepted, user_prob, accept_p = user_model.evaluate_recommendation(
                item_id=action,
                candidates=candidates,
                mode="candidate_set",
            )
        else:
            raise ValueError(
                f"Unknown acceptance_mode={acceptance_mode!r}. "
                "Use 'direct' or 'candidate_set'."
            )

        reward = 1.0 if accepted else 0.0

        if accepted:
            next_item = action
        else:
            next_item = user_model.sample_next_item(
                exclude=list(consumed_items),
                allowed_items=valid_allowed,
            )

        user_model.step(next_item)
        consumed_items.add(next_item)

        episode.log_step(
            recommended_item=action,
            accepted=accepted,
            reward=reward,
            user_prob=user_prob,
            accept_probability=accept_p,
            catalog_size=len(valid_allowed),
        )

    return episode


def run_simulation(
    user_model,
    agent,
    sessions,
    allowed_items,
    warmup_length=5,
    horizon=10,
    forbid_repeated=True,
    acceptance_mode="direct",
    num_candidates=100,
):
    sim_metrics = SimulationMetrics()

    for seq in sessions:
        if len(seq) <= warmup_length:
            continue

        warmup = seq[:warmup_length]

        episode = run_episode(
            user_model=user_model,
            agent=agent,
            warmup_items=warmup,
            allowed_items=allowed_items,
            horizon=horizon,
            forbid_repeated=forbid_repeated,
            acceptance_mode=acceptance_mode,
            num_candidates=num_candidates,
        )

        sim_metrics.add_episode(episode)

    return sim_metrics.compute()