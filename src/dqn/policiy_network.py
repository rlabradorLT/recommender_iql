import torch
import torch.nn as nn
    
class PolicyNet(nn.Module):

    def __init__(self, state_dim, num_actions, h1, h2):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, num_actions)
        )

        # stable init
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, s):
        return self.net(s)
