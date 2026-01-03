import torch
import numpy as np
import time
import gymnasium as gym

from envs.gym_wrapper import GymWrapper
from agents.dqn_agent import DQNAgent

# -------------------------------------------------
# Device
# -------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------------------------
# Environment (RENDER ENABLED)
# -------------------------------------------------
env = GymWrapper(env_name="CartPole-v1")
env.env = gym.make("CartPole-v1", render_mode="human")

# -------------------------------------------------
# Minimal config REQUIRED by DQNAgent
# -------------------------------------------------
eval_cfg = {
    "lr": 0.001,
    "gamma": 0.99,
    "tau": 0.001,
    "buffer_size": 1,
    "batch_size": 1
}

# -------------------------------------------------
# Agent
# -------------------------------------------------
agent = DQNAgent(
    state_size=env.state_size,
    action_size=env.action_size,
    cfg=eval_cfg,
    device=device
)

# -------------------------------------------------
# Load trained model
# -------------------------------------------------
agent.q_local.load_state_dict(
    torch.load("checkpoints/dqn_50_episodes.pth", map_location=device)
)
agent.q_local.eval()

print("\n🎮 Evaluating trained DQN agent (ε = 0)\n")

# -------------------------------------------------
# Evaluation loop
# -------------------------------------------------
NUM_EVAL_EPISODES = 5
scores = []

for ep in range(NUM_EVAL_EPISODES):
    state = env.reset()
    done = False
    total_reward = 0.0

    while not done:
        action = agent.act(state, eps=0.0)
        next_state, reward, done = env.step(action)

        state = next_state
        total_reward += reward
        time.sleep(0.02)

    scores.append(total_reward)
    print(f"Episode {ep + 1}: Score = {total_reward}")

env.close()

# -------------------------------------------------
# Summary
# -------------------------------------------------
print("\n📊 Evaluation Summary")
print(f"Average Score: {np.mean(scores):.2f}")
print(f"Max Score:     {np.max(scores):.2f}")
print(f"Min Score:     {np.min(scores):.2f}")
