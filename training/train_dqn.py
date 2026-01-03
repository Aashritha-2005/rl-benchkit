import gymnasium as gym
import torch
import numpy as np
import os

from agents.dqn_agent import DQNAgent

# -------------------------------------------------
# Device
# -------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------------------------
# Environment (benchmark only)
# -------------------------------------------------
env = gym.make("CartPole-v1")

state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# -------------------------------------------------
# DQN Config
# -------------------------------------------------
cfg = {
    "lr": 1e-3,
    "gamma": 0.99,
    "tau": 1e-3,
    "buffer_size": 10000,
    "batch_size": 64,
    "epsilon_start": 1.0,
    "epsilon_end": 0.01,
    "epsilon_decay": 0.995
}

# -------------------------------------------------
# Agent
# -------------------------------------------------
agent = DQNAgent(
    state_size=state_size,
    action_size=action_size,
    cfg=cfg,
    device=device
)

# -------------------------------------------------
# Training controls (HARD STOP)
# -------------------------------------------------
TOTAL_EPISODES = 50
MAX_STEPS = 1000

print("\n[INIT] DQN training started")
print("[INFO] Environment: CartPole-v1 (benchmark only)")
print("[INFO] Episodes:", TOTAL_EPISODES)
print("[INFO] Hard stop enabled\n")

epsilon = cfg["epsilon_start"]

# -------------------------------------------------
# Training loop (CORRECT)
# -------------------------------------------------
for episode in range(1, TOTAL_EPISODES + 1):

    state, _ = env.reset()
    episode_return = 0.0

    for _ in range(MAX_STEPS):
        action = agent.act(state, eps=epsilon)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        agent.step(state, action, reward, next_state, done)

        state = next_state
        episode_return += reward

        if done:
            break

    epsilon = max(cfg["epsilon_end"], epsilon * cfg["epsilon_decay"])

    print(f"[TRAIN] DQN | Episode {episode}/{TOTAL_EPISODES} | Return: {episode_return:.1f} | Epsilon: {epsilon:.3f}")

# -------------------------------------------------
# Save model
# -------------------------------------------------
os.makedirs("checkpoints", exist_ok=True)
torch.save(agent.q_local.state_dict(), "checkpoints/dqn_cartpole.pth")

print("\n[DONE] DQN training complete")
env.close()
