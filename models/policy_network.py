import torch.nn as nn

class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU()
        )
        self.policy = nn.Linear(128, action_size)
        self.value = nn.Linear(128, 1)

    def forward(self, x):
        h = self.shared(x)
        return self.policy(h), self.value(h)
