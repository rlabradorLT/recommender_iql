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

def _to_tensor_state(state: np.ndarray) -> torch.Tensor:
    """
    Convierte un estado numpy 1D a tensor batch [1, obs_dim].
    """
    if not isinstance(state, np.ndarray):
        state = np.asarray(state, dtype=np.float32)

    state = state.astype(np.float32, copy=False)

    if state.ndim != 1:
        raise ValueError(f"Expected 1D state, got shape {state.shape}")

    return torch.from_numpy(state).to(DEVICE).unsqueeze(0)


def _mask_forbidden(scores: np.ndarray, forbidden_items=None) -> np.ndarray:
    """
    Enmascara items prohibidos asignando -inf.
    """
    out = scores.copy()

    if forbidden_items is None:
        return out

    if len(forbidden_items) == 0:
        return out

    idx = np.asarray(list(forbidden_items), dtype=np.int64)

    idx = idx[(idx >= 0) & (idx < len(out))]
    out[idx] = -np.inf

    return out


# ============================================================
# agente base
# ============================================================

class BaseAgent:
    """
    Interfaz común para todos los agentes del simulador.
    """

    def score(self, state, forbidden_items=None) -> np.ndarray:
        raise NotImplementedError

    def recommend(self, state, forbidden_items=None) -> int:
        scores = self.score(state, forbidden_items=forbidden_items)

        if scores.ndim != 1:
            raise ValueError(f"Scores must be 1D, got shape {scores.shape}")

        if not np.isfinite(scores).any():
            raise RuntimeError("All agent scores are non-finite after masking.")

        return int(np.argmax(scores))


# ============================================================
# agentes basados en policy
# ============================================================

class PolicyAgent(BaseAgent):
    """
    Wrapper para BC e IQL Policy.
    """

    def __init__(self, checkpoint_path: str, stats_path: str):

        ckpt = torch.load(checkpoint_path, map_location=DEVICE)

        self.model = PolicyNet(
            obs_dim=ckpt["obs_dim"],
            num_items=ckpt["num_items"],
            hidden1=ckpt["hidden1"],
            hidden2=ckpt["hidden2"],
        ).to(DEVICE)

        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        self.num_items = ckpt["num_items"]
        self.obs_dim = ckpt["obs_dim"]

        self.norm_stats = load_normalization_stats(stats_path)

    @torch.no_grad()
    def score(self, state, forbidden_items=None) -> np.ndarray:

        state = normalize_obs(state, self.norm_stats)
        state_t = _to_tensor_state(state)

        logits = self.model(state_t).squeeze(0).detach().cpu().numpy()

        if logits.shape[0] != self.num_items:
            raise ValueError(
                f"Expected {self.num_items} item scores, got {logits.shape[0]}"
            )

        logits = _mask_forbidden(logits, forbidden_items=forbidden_items)

        return logits


# ============================================================
# agentes basados en critic
# ============================================================

class CriticAgent(BaseAgent):
    """
    Wrapper para IQL critic ranking y CQL critic ranking.
    """

    def __init__(self, checkpoint_path: str, stats_path: str):

        ckpt = torch.load(checkpoint_path, map_location=DEVICE)

        self.q1 = QNetwork(
            obs_dim=ckpt["obs_dim"],
            num_items=ckpt["num_items"],
            hidden1=ckpt["hidden1"],
            hidden2=ckpt["hidden2"],
        ).to(DEVICE)

        self.q2 = QNetwork(
            obs_dim=ckpt["obs_dim"],
            num_items=ckpt["num_items"],
            hidden1=ckpt["hidden1"],
            hidden2=ckpt["hidden2"],
        ).to(DEVICE)

        self.q1.load_state_dict(ckpt["q1_state_dict"])
        self.q2.load_state_dict(ckpt["q2_state_dict"])

        self.q1.eval()
        self.q2.eval()

        self.num_items = ckpt["num_items"]
        self.obs_dim = ckpt["obs_dim"]

        self.norm_stats = load_normalization_stats(stats_path)

    @torch.no_grad()
    def score(self, state, forbidden_items=None) -> np.ndarray:

        state = normalize_obs(state, self.norm_stats)
        state_t = _to_tensor_state(state)

        q1 = self.q1(state_t)
        q2 = self.q2(state_t)

        q = torch.min(q1, q2).squeeze(0).detach().cpu().numpy()

        if q.shape[0] != self.num_items:
            raise ValueError(
                f"Expected {self.num_items} item scores, got {q.shape[0]}"
            )

        q = _mask_forbidden(q, forbidden_items=forbidden_items)

        return q


# ============================================================
# factory
# ============================================================

def load_agent(agent_type: str, checkpoint_path: str, stats_path: str) -> BaseAgent:
    """
    Carga un agente del simulador con interfaz común.

    agent_type:
        - bc
        - iql_policy
        - iql_critic
        - cql
    """

    agent_type = agent_type.lower().strip()

    if agent_type in {"bc", "iql_policy"}:
        return PolicyAgent(checkpoint_path=checkpoint_path, stats_path=stats_path)

    if agent_type in {"iql_critic", "cql"}:
        return CriticAgent(checkpoint_path=checkpoint_path, stats_path=stats_path)

    raise ValueError(f"Unknown agent type: {agent_type}")