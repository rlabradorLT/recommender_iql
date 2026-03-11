import torch
import torch.nn.functional as F
import numpy as np

from dataset_pipeline.gru_model import GRUEncoder, NextItemHead


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class GRUUserModel:
    """
    Simulador de usuario basado en GRU.

    Modela:

        p(item | history)

    usando:

        history -> GRU -> hidden_state -> head -> logits -> softmax
    """

    def __init__(
        self,
        checkpoint_path: str,
        temperature: float = 1.0,
        min_prob: float = 1e-8,
    ):

        ckpt = torch.load(checkpoint_path, map_location=DEVICE)

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

        self.hid_dim = ckpt["hid_dim"]
        self.num_items = ckpt["num_items"]

        self.temperature = temperature
        self.min_prob = min_prob

        self.reset()

    # ---------------------------------------------------------
    # estado del usuario
    # ---------------------------------------------------------

    def reset(self):

        self.h = torch.zeros(1, 1, self.hid_dim, device=DEVICE)

    def warmup(self, items):

        for item in items:

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

        probs = F.softmax(logits, dim=-1)

        probs = probs.squeeze(0)

        probs = torch.clamp(probs, min=self.min_prob)

        probs = probs / probs.sum()

        return probs

    # ---------------------------------------------------------
    # aceptación recomendación
    # ---------------------------------------------------------

    @torch.no_grad()
    def evaluate_recommendation(self, item_id: int):

        probs = self.distribution()

        p = probs[item_id].item()

        accepted = np.random.rand() < p

        return accepted, p

    # ---------------------------------------------------------
    # muestrear siguiente item
    # ---------------------------------------------------------

    @torch.no_grad()
    def sample_next_item(self, exclude=None):

        probs = self.distribution().clone()

        if exclude is not None and len(exclude) > 0:

            probs[exclude] = 0.0

            if probs.sum() <= 0:
                probs = self.distribution()

        probs = probs / probs.sum()

        item = torch.multinomial(probs, 1).item()

        return item

    # ---------------------------------------------------------
    # actualizar estado usuario
    # ---------------------------------------------------------

    def step(self, item_id):

        item_tensor = torch.tensor([[item_id]], device=DEVICE)

        _, self.h = self.encoder(item_tensor, self.h)