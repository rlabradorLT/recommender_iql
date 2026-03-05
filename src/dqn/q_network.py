import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Outputs Q(s) for ALL discrete actions: shape (B, num_actions)."""
    def __init__(self, state_dim: int, num_actions: int, h1: int, h2: int):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, num_actions)

        # Stable init:
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity="relu")
        nn.init.zeros_(self.fc1.bias)
        nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity="relu")
        nn.init.zeros_(self.fc2.bias)

        # CRITICAL: start Q near 0
        nn.init.zeros_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(s))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

