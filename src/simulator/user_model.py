import torch
import torch.nn.functional as F
import numpy as np

from dataset_pipeline.gru_model import GRUEncoder, NextItemHead


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class GRUUserModel:
    """
    Simulador de usuario basado en GRU.

    Modela una distribución de siguiente ítem:

        p(item | history)

    y la convierte en:
    - probabilidad de aceptación de una recomendación
    - distribución de transición cuando el usuario no acepta
    """

    def __init__(
        self,
        checkpoint_path: str,
        temperature: float = 1.0,
        min_prob: float = 1e-8,
        acceptance_scale: float = 50.0,
        acceptance_min: float = 0.01,
        acceptance_max: float = 0.99,
    ):

        ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)

        self.encoder = GRUEncoder(
            num_items=ckpt["num_items"],
            emb_dim=ckpt["emb_dim"],
            hid_dim=ckpt["hid_dim"],
            pad_item_id=ckpt.get("pad_item_id", 0),
        ).to(DEVICE)

        self.head = NextItemHead(
            hid_dim=ckpt["hid_dim"],
            num_items=ckpt["num_items"],
        ).to(DEVICE)

        self.encoder.load_state_dict(ckpt["encoder_state_dict"])
        self.head.load_state_dict(ckpt["head_state_dict"])

        self.encoder.eval()
        self.head.eval()

        self.hid_dim = int(ckpt["hid_dim"])
        self.num_items = int(ckpt["num_items"])

        self.temperature = float(temperature)
        self.min_prob = float(min_prob)

        # Parámetros para mapear p(item|history) -> p(accept)
        self.acceptance_scale = float(acceptance_scale)
        self.acceptance_min = float(acceptance_min)
        self.acceptance_max = float(acceptance_max)

        if self.temperature <= 0:
            raise ValueError("temperature must be > 0")

        if not (0.0 <= self.acceptance_min <= self.acceptance_max <= 1.0):
            raise ValueError(
                "acceptance_min and acceptance_max must satisfy "
                "0 <= min <= max <= 1"
            )

        self.reset()

    # ---------------------------------------------------------
    # estado del usuario
    # ---------------------------------------------------------

    def reset(self):
        self.h = torch.zeros(1, 1, self.hid_dim, device=DEVICE)

    def warmup(self, items):
        for item in items:
            item = int(item)

            if not (0 <= item < self.num_items):
                raise ValueError(
                    f"Warmup item {item} is outside user-model range [0, {self.num_items})"
                )

            item_tensor = torch.tensor([[item]], device=DEVICE)
            _, self.h = self.encoder(item_tensor, self.h)

    def get_state(self):
        return self.h[0, 0].detach().cpu().numpy()

    # ---------------------------------------------------------
    # distribución del usuario
    # ---------------------------------------------------------

    @torch.no_grad()
    def distribution(self):
        logits = self.head(self.h[0])
        logits = logits / self.temperature

        probs = F.softmax(logits, dim=-1).squeeze(0)
        probs = torch.clamp(probs, min=self.min_prob)
        probs = probs / probs.sum()

        return probs

    # ---------------------------------------------------------
    # utilidades internas
    # ---------------------------------------------------------

    def _validate_item_id(self, item_id: int):
        item_id = int(item_id)

        if not (0 <= item_id < self.num_items):
            raise ValueError(
                f"item_id={item_id} is outside user-model range [0, {self.num_items})"
            )

        return item_id

    def _normalize_probability_vector(self, probs: torch.Tensor) -> torch.Tensor:
        total = probs.sum()

        if total <= 0:
            raise RuntimeError("Probability vector has non-positive mass.")

        return probs / total

    def _direct_accept_probability(self, item_prob: float) -> float:
        """
        Convierte p(item|history) en una probabilidad de aceptación más usable.

        Motivación:
        en catálogos grandes, p(item|history) suele ser muy pequeña incluso
        para ítems razonables. Si usamos esa probabilidad en bruto, casi nunca
        habría aceptación.

        Por eso usamos un reescalado lineal + clipping.
        """
        accept_p = item_prob * self.acceptance_scale
        accept_p = float(np.clip(accept_p, self.acceptance_min, self.acceptance_max))
        return accept_p

    def _candidate_set_accept_probability(
        self,
        item_id: int,
        candidates: np.ndarray,
        probs: torch.Tensor,
    ) -> float:
        """
        Mantiene el esquema por ranking relativo para comparaciones controladas.
        """
        if candidates is None:
            raise ValueError("candidates must be provided in candidate_set mode.")

        candidates = np.asarray(candidates, dtype=np.int64)

        if candidates.ndim != 1 or candidates.size == 0:
            raise ValueError("candidates must be a non-empty 1D array.")

        valid_mask = (candidates >= 0) & (candidates < self.num_items)
        candidates = candidates[valid_mask]

        if candidates.size == 0:
            raise ValueError("All candidates are invalid under the user model.")

        if item_id not in set(candidates.tolist()):
            raise ValueError("Recommended item must be included in candidates.")

        cand_probs = probs[candidates]
        ranking = torch.argsort(cand_probs, descending=True)

        rec_index = int(np.where(candidates == item_id)[0][0])
        rank = int(torch.where(ranking == rec_index)[0].item())

        if rank < 3:
            return 0.9
        if rank < 10:
            return 0.6
        if rank < 20:
            return 0.3
        return 0.05

    # ---------------------------------------------------------
    # aceptación recomendación
    # ---------------------------------------------------------

    @torch.no_grad()
    def evaluate_recommendation(self, item_id, candidates=None, mode="direct"):
        item_id = self._validate_item_id(item_id)

        probs = self.distribution()
        item_prob = float(probs[item_id].item())

        if mode == "direct":
            accept_p = self._direct_accept_probability(item_prob)

        elif mode == "candidate_set":
            accept_p = self._candidate_set_accept_probability(
                item_id=item_id,
                candidates=candidates,
                probs=probs,
            )

        else:
            raise ValueError(
                f"Unknown mode={mode!r}. Use 'direct' or 'candidate_set'."
            )

        accepted = bool(np.random.rand() < accept_p)

        return accepted, item_prob, float(accept_p)

    # ---------------------------------------------------------
    # muestrear siguiente item
    # ---------------------------------------------------------

    @torch.no_grad()
    def sample_next_item(self, exclude=None, allowed_items=None):
        probs = self.distribution().clone()

        # Restringir a catálogo permitido del simulador
        if allowed_items is not None:
            allowed_items = np.asarray(list(set(allowed_items)), dtype=np.int64)
            allowed_items = allowed_items[
                (allowed_items >= 0) & (allowed_items < self.num_items)
            ]

            restricted = torch.zeros_like(probs)

            if allowed_items.size > 0:
                restricted[allowed_items] = probs[allowed_items]
                probs = restricted
            else:
                raise ValueError("allowed_items has no valid ids for the user model.")

        # Excluir items ya consumidos si se desea
        if exclude is not None and len(exclude) > 0:
            exclude = np.asarray(list(set(exclude)), dtype=np.int64)
            exclude = exclude[(exclude >= 0) & (exclude < self.num_items)]
            probs[exclude] = 0.0

        # Fallback: si se agotó toda la masa por restricciones, relajar exclusión
        if probs.sum() <= 0:
            probs = self.distribution().clone()

            if allowed_items is not None:
                restricted = torch.zeros_like(probs)
                restricted[allowed_items] = probs[allowed_items]
                probs = restricted

            if probs.sum() <= 0:
                raise RuntimeError(
                    "No probability mass left after applying allowed_items."
                )

        probs = self._normalize_probability_vector(probs)

        item = int(torch.multinomial(probs, 1).item())
        return item

    # ---------------------------------------------------------
    # actualizar estado usuario
    # ---------------------------------------------------------

    def step(self, item_id):
        item_id = self._validate_item_id(item_id)

        item_tensor = torch.tensor([[item_id]], device=DEVICE)
        _, self.h = self.encoder(item_tensor, self.h)