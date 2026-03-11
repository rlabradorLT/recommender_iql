import numpy as np

from simulator.metrics import EpisodeMetrics, SimulationMetrics

def sample_candidates(action, num_items, num_negatives=99):
    """
    Genera candidate set: 1 positivo + N negativos.
    Usa la misma lógica que build_candidates.py
    """

    negs = np.random.randint(0, num_items, size=num_negatives)

    mask = (negs == action)
    while mask.any():
        negs[mask] = np.random.randint(0, num_items, size=mask.sum())
        mask = (negs == action)

    return np.concatenate(([action], negs))

def run_episode(
    user_model,
    agent,
    warmup_items,
    allowed_items,
    horizon=10,
    forbid_repeated=True,
    num_candidates=100,
):

    episode = EpisodeMetrics()

    user_model.reset()
    user_model.warmup(warmup_items)

    consumed_items = set(warmup_items)

    for _ in range(horizon):

        state = user_model.get_state()

        scores = agent.score(state)

        # ----------------------------------------------
        # restringir catálogo a small matrix
        # ----------------------------------------------

        mask = np.full_like(scores, -np.inf)

        allowed = list(allowed_items)

        mask[allowed] = scores[allowed]

        scores = mask

        # ----------------------------------------------
        # evitar repetir items
        # ----------------------------------------------

        if forbid_repeated:

            for item in consumed_items:
                if item < len(scores):
                    scores[item] = -np.inf

        # action = int(np.argmax(scores))
        # -----------------------------------------------------
        # selección acción (epsilon-greedy sobre top-k)
        # -----------------------------------------------------

        # epsilon = 0.05
        # top_k = 20

        # top_items = np.argsort(scores)[-top_k:]

        # if np.random.rand() < epsilon:
        #     action = int(np.random.choice(top_items))
        # else:
        #     action = int(top_items[np.argmax(scores[top_items])])
            
        # accepted, user_prob = user_model.evaluate_recommendation(action)
        action = int(np.argmax(scores))

        candidates = sample_candidates(
            action,
            len(scores),
            num_negatives=num_candidates - 1
        )

        accepted, user_prob = user_model.evaluate_recommendation(
            action,
            candidates
        )
        
        reward = 1.0 if accepted else 0.0

        if accepted:
            next_item = action
        else:
            next_item = user_model.sample_next_item(
                exclude=list(consumed_items)
            )

        user_model.step(next_item)

        consumed_items.add(next_item)

        episode.log_step(
            recommended_item=action,
            accepted=accepted,
            reward=reward,
            user_prob=user_prob,
        )

    return episode


def run_simulation(
    user_model,
    agent,
    sessions,
    allowed_items,
    warmup_length=5,
    horizon=10,
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
            num_candidates=num_candidates,
        )

        sim_metrics.add_episode(episode)

    return sim_metrics.compute()