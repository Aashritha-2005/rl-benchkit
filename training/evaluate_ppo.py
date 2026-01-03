import torch
import numpy as np
import time
import gymnasium as gym

from agents.ppo_agent import PPOAgent

# -------------------------------------------------
# Device
# -------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------------------------
# Environment (render ON)
# -------------------------------------------------
env = gym.make("CartPole-v1", render_mode="human")

state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# -------------------------------------------------
# PPO Config (same as training)
# -------------------------------------------------
ppo_cfg = {
    "lr": 3e-4,
    "gamma": 0.99,
    "clip_eps": 0.2
}

# -------------------------------------------------
# Agent
# -------------------------------------------------
agent = PPOAgent(
    state_size=state_size,
    action_size=action_size,
    cfg=ppo_cfg,
    device=device
)

# -------------------------------------------------
# Load trained model
# -------------------------------------------------
agent.policy.load_state_dict(
    torch.load("checkpoints/ppo_cartpole.pth", map_location=device)
)
agent.policy.eval()

print("\n🎮 Evaluating PPO agent (deterministic policy)\n")

# -------------------------------------------------
# Evaluation loop
# -------------------------------------------------
NUM_EPISODES = 5
scores = []

for ep in range(NUM_EPISODES):
    state, _ = env.reset()
    done = False
    episode_reward = 0

    while not done:
        with torch.no_grad():
            action, _, _ = agent.act(state)

        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        episode_reward += reward
        time.sleep(0.02)

    scores.append(episode_reward)
    print(f"Episode {ep + 1}: Score = {episode_reward}")

env.close()

# -------------------------------------------------
# Summary
# -------------------------------------------------
print("\n📊 PPO Evaluation Summary")
print(f"Average Score: {np.mean(scores):.2f}")
print(f"Max Score:     {np.max(scores):.2f}")
print(f"Min Score:     {np.min(scores):.2f}")
