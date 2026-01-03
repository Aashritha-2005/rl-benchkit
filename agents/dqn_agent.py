import torch
import torch.nn.functional as F
from models.dueling_q_network import DuelingQNetwork
from memory.replay_buffer import ReplayBuffer

class DQNAgent:
    def __init__(self, state_size, action_size, cfg, device):
        self.device = device
        self.q_local = DuelingQNetwork(state_size, action_size).to(device)
        self.q_target = DuelingQNetwork(state_size, action_size).to(device)
        self.optimizer = torch.optim.Adam(self.q_local.parameters(), lr=cfg["lr"])
        self.memory = ReplayBuffer(cfg["buffer_size"], cfg["batch_size"], device)
        self.gamma = cfg["gamma"]
        self.tau = cfg["tau"]
        self.action_size = action_size

    def act(self, state, eps):
        if torch.rand(1).item() < eps:
            return torch.randint(0, self.action_size, (1,)).item()
        state = torch.tensor(state).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.q_local(state).argmax().item()

    def step(self, s, a, r, ns, d):
        self.memory.add(s, a, r, ns, d)
        if len(self.memory) > self.memory.batch_size:
            self.learn()

    def learn(self):
        s, a, r, ns, d = self.memory.sample()
        next_a = self.q_local(ns).argmax(1).unsqueeze(1)
        q_next = self.q_target(ns).gather(1, next_a)
        q_target = r + self.gamma * q_next * (1 - d)
        q_expected = self.q_local(s).gather(1, a)

        loss = F.mse_loss(q_expected, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        for t, l in zip(self.q_target.parameters(), self.q_local.parameters()):
            t.data.copy_(self.tau * l.data + (1 - self.tau) * t.data)
