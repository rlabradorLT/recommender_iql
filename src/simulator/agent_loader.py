import numpy as np
import torch

from dqn.policiy_network import PolicyNet
from dqn.q_network import QNetwork
from dataset_pipeline.normalization import (
    load_normalization_stats,
    normalize_obs,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# utilidades
# ============================================================

def _safe_torch_load(path: str):
    return torch.load(path, map_location=DEVICE, weights_only=False)


def _to_tensor_state(state: np.ndarray) -> torch.Tensor:
    if not isinstance(state, np.ndarray):
        state = np.asarray(state, dtype=np.float32)

    state = state.astype(np.float32, copy=False)

    if state.ndim == 2 and state.shape[0] == 1:
        state = state[0]

    if state.ndim != 1:
        raise ValueError(f"Expected 1D state, got shape {state.shape}")

    return torch.from_numpy(state).to(DEVICE).unsqueeze(0)


def _mask_forbidden(scores: np.ndarray, forbidden_items=None) -> np.ndarray:
    out = np.asarray(scores, dtype=np.float32).copy()

    if forbidden_items is None:
        return out

    if len(forbidden_items) == 0:
        return out

    idx = np.asarray(list(forbidden_items), dtype=np.int64)
    idx = idx[(idx >= 0) & (idx < len(out))]
    out[idx] = -np.inf

    return out


def _finite_values(scores: np.ndarray) -> np.ndarray:
    finite = scores[np.isfinite(scores)]

    if finite.size == 0:
        raise RuntimeError("All scores are non-finite.")

    return finite


def _transform_scores(scores: np.ndarray, transform: str) -> np.ndarray:
    """
    Transformación monotónica por estado.

    Objetivo:
    - hacer explícito el pipeline de decisión;
    - permitir auditoría/control;
    - sin alterar el orden relativo útil cuando no corresponde.

    NOTA:
    'rank' y 'zscore' se aplican solo sobre valores finitos.
    """
    scores = np.asarray(scores, dtype=np.float32).copy()
    transform = str(transform).lower().strip()

    finite_mask = np.isfinite(scores)

    if not finite_mask.any():
        raise RuntimeError("All scores are non-finite before transform.")

    finite_vals = scores[finite_mask]

    if transform == "none":
        return scores

    if transform == "zscore":
        mean = float(finite_vals.mean())
        std = float(finite_vals.std())

        if std < 1e-12:
            scores[finite_mask] = 0.0
        else:
            scores[finite_mask] = (finite_vals - mean) / std

        return scores

    if transform == "rank":
        order = np.argsort(finite_vals)
        ranks = np.empty_like(order, dtype=np.float32)
        ranks[order] = np.arange(len(finite_vals), dtype=np.float32)
        scores[finite_mask] = ranks
        return scores

    raise ValueError(
        f"Unknown score_transform={transform!r}. "
        "Use 'none', 'zscore', or 'rank'."
    )


def _get_state_dim(ckpt: dict, stats_path: str) -> int:
    if "state_dim" in ckpt:
        return int(ckpt["state_dim"])

    mean, _ = load_normalization_stats(stats_path)
    return int(mean.shape[-1])


def _get_num_actions(ckpt: dict, state_dict: dict) -> int:
    if "num_actions" in ckpt:
        return int(ckpt["num_actions"])

    if "num_items" in ckpt:
        return int(ckpt["num_items"])

    # PolicyNet final layer
    if "net.4.weight" in state_dict:
        return int(state_dict["net.4.weight"].shape[0])

    # QNetwork final layer
    if "fc3.weight" in state_dict:
        return int(state_dict["fc3.weight"].shape[0])

    raise KeyError("Could not infer num_actions / num_items from checkpoint.")


def _infer_policy_hidden_dims(state_dict: dict):
    # PolicyNet:
    # net.0 = Linear(state_dim, h1)
    # net.2 = Linear(h1, h2)
    # net.4 = Linear(h2, num_actions)
    h1 = int(state_dict["net.0.weight"].shape[0])
    h2 = int(state_dict["net.2.weight"].shape[0])
    return h1, h2


def _infer_q_hidden_dims(state_dict: dict):
    # QNetwork:
    # fc1 = Linear(state_dim, h1)
    # fc2 = Linear(h1, h2)
    # fc3 = Linear(h2, num_actions)
    h1 = int(state_dict["fc1.weight"].shape[0])
    h2 = int(state_dict["fc2.weight"].shape[0])
    return h1, h2


def _extract_policy_state_dict(ckpt: dict):
    if "policy" in ckpt:
        return ckpt["policy"]

    if "model_state_dict" in ckpt:
        return ckpt["model_state_dict"]

    raise KeyError("Policy checkpoint does not contain 'policy' or 'model_state_dict'.")


def _extract_q_state_dicts(ckpt: dict):
    if "Q1" in ckpt and "Q2" in ckpt:
        return ckpt["Q1"], ckpt["Q2"]

    if "q1_state_dict" in ckpt and "q2_state_dict" in ckpt:
        return ckpt["q1_state_dict"], ckpt["q2_state_dict"]

    raise KeyError("Critic checkpoint does not contain Q1/Q2 weights.")


# ============================================================
# agente base
# ============================================================

class BaseAgent:

    def __init__(self, score_transform: str = "none"):
        self.score_transform = score_transform
        self.score_type = "unknown"

    def raw_score(self, state) -> np.ndarray:
        raise NotImplementedError

    def score(self, state, forbidden_items=None) -> np.ndarray:
        scores = self.raw_score(state)

        if scores.ndim != 1:
            raise ValueError(f"Scores must be 1D, got shape {scores.shape}")

        if scores.shape[0] != self.num_actions:
            raise ValueError(
                f"Expected {self.num_actions} scores, got {scores.shape[0]}"
            )

        scores = _mask_forbidden(scores, forbidden_items=forbidden_items)
        scores = _transform_scores(scores, transform=self.score_transform)

        return scores

    def recommend(self, state, forbidden_items=None):

        scores = self.score(state, forbidden_items)

        temperature = 0.05

        probs = np.exp(scores / temperature)
        probs = probs / probs.sum()

        return int(np.random.choice(len(scores), p=probs))

    def describe(self) -> dict:
        return {
            "score_type": self.score_type,
            "score_transform": self.score_transform,
            "num_actions": int(self.num_actions),
            "state_dim": int(self.state_dim),
        }


# ============================================================
# agentes policy
# ============================================================

class PolicyAgent(BaseAgent):

    def __init__(
        self,
        checkpoint_path: str,
        stats_path: str,
        score_transform: str = "none",
    ):
        super().__init__(score_transform=score_transform)

        ckpt = _safe_torch_load(checkpoint_path)

        state_dict = _extract_policy_state_dict(ckpt)
        state_dim = _get_state_dim(ckpt, stats_path)
        num_actions = _get_num_actions(ckpt, state_dict)
        hidden1, hidden2 = _infer_policy_hidden_dims(state_dict)

        self.model = PolicyNet(
            state_dim,
            num_actions,
            hidden1,
            hidden2,
        ).to(DEVICE)

        self.model.load_state_dict(state_dict)
        self.model.eval()

        self.num_actions = num_actions
        self.state_dim = state_dim
        self.norm_stats = load_normalization_stats(stats_path)
        self.score_type = "policy_logit"

    @torch.no_grad()
    def raw_score(self, state) -> np.ndarray:
        state = normalize_obs(state, self.norm_stats)
        state_t = _to_tensor_state(state)

        logits = self.model(state_t).squeeze(0).detach().cpu().numpy()
        return np.asarray(logits, dtype=np.float32)


# ============================================================
# agentes critic
# ============================================================

class CriticAgent(BaseAgent):

    def __init__(
        self,
        checkpoint_path: str,
        stats_path: str,
        score_transform: str = "none",
    ):
        super().__init__(score_transform=score_transform)

        ckpt = _safe_torch_load(checkpoint_path)

        q1_state, q2_state = _extract_q_state_dicts(ckpt)
        state_dim = _get_state_dim(ckpt, stats_path)
        num_actions = _get_num_actions(ckpt, q1_state)
        hidden1, hidden2 = _infer_q_hidden_dims(q1_state)

        self.q1 = QNetwork(
            state_dim,
            num_actions,
            hidden1,
            hidden2,
        ).to(DEVICE)

        self.q2 = QNetwork(
            state_dim,
            num_actions,
            hidden1,
            hidden2,
        ).to(DEVICE)

        self.q1.load_state_dict(q1_state)
        self.q2.load_state_dict(q2_state)

        self.q1.eval()
        self.q2.eval()

        self.num_actions = num_actions
        self.state_dim = state_dim
        self.norm_stats = load_normalization_stats(stats_path)
        self.score_type = "critic_qmin"

    @torch.no_grad()
    def raw_score(self, state) -> np.ndarray:
        state = normalize_obs(state, self.norm_stats)
        state_t = _to_tensor_state(state)

        q1 = self.q1(state_t)
        q2 = self.q2(state_t)

        q = torch.minimum(q1, q2).squeeze(0).detach().cpu().numpy()
        return np.asarray(q, dtype=np.float32)


# ============================================================
# factory
# ============================================================

def load_agent(
    agent_type: str,
    checkpoint_path: str,
    stats_path: str,
    score_transform: str = "none",
) -> BaseAgent:

    agent_type = agent_type.lower().strip()

    if agent_type in {"bc", "iql_policy"}:
        return PolicyAgent(
            checkpoint_path=checkpoint_path,
            stats_path=stats_path,
            score_transform=score_transform,
        )

    if agent_type in {"iql_critic", "cql"}:
        return CriticAgent(
            checkpoint_path=checkpoint_path,
            stats_path=stats_path,
            score_transform=score_transform,
        )

    raise ValueError(f"Unknown agent type: {agent_type}")