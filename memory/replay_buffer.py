import random
import numpy as np
import torch
from collections import deque

class ReplayBuffer:
    def __init__(self, size, batch_size, device):
        self.memory = deque(maxlen=size)
        self.batch_size = batch_size
        self.device = device

    def add(self, s, a, r, ns, d):
        self.memory.append((s, a, r, ns, d))

    def sample(self):
        batch = random.sample(self.memory, self.batch_size)
        s, a, r, ns, d = zip(*batch)
        return (
            torch.tensor(s).float().to(self.device),
            torch.tensor(a).long().unsqueeze(1).to(self.device),
            torch.tensor(r).float().unsqueeze(1).to(self.device),
            torch.tensor(ns).float().to(self.device),
            torch.tensor(d).float().unsqueeze(1).to(self.device),
        )

    def __len__(self):
        return len(self.memory)
