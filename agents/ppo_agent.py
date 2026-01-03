import torch
import torch.nn.functional as F
from models.policy_network import PolicyNetwork

class PPOAgent:
    def __init__(self, state_size, action_size, cfg, device):
        self.policy = PolicyNetwork(state_size, action_size).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=cfg["lr"])
        self.clip = cfg["clip_eps"]
        self.gamma = cfg["gamma"]
        self.device = device

    def act(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)

        mu, log_std = self.policy(state)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mu, std)


        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)

        action_clipped = action.clamp(-2.0, 2.0)
        return action_clipped.cpu().numpy()[0], log_prob


    def learn(self, states, actions, old_log_probs, returns):

        for _ in range(4):  # PPO epochs

            logits, values = self.policy(states)
            mu, log_std = self.policy(states)
            std = torch.exp(log_std)

            dist = torch.distributions.Normal(mu, std)
            log_probs = dist.log_prob(actions).sum(dim=-1)

            ratio = (log_probs - old_log_probs).exp()

            # Advantage
            adv = returns - values.squeeze()
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            # Entropy bonus
            entropy = dist.entropy().mean()

            loss = -torch.min(
                ratio * adv,
                torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * adv
            ).mean() - 0.01 * entropy

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
