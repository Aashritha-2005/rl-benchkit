import gymnasium as gym
import torch
import numpy as np
import os

from agents.ppo_agent import PPOAgent

# -------------------------------------------------
# Episode returns (for analysis)
# -------------------------------------------------
episode_returns = []

# -------------------------------------------------
# Device
# -------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------------------------
# Environment (FINAL DECISION)
# -------------------------------------------------
ENV_NAME = "Pendulum-v1"
env = gym.make(ENV_NAME)

state_size = env.observation_space.shape[0]
action_size = env.action_space.shape[0]

# -------------------------------------------------
# PPO Config
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
# Training Controls
# -------------------------------------------------
TOTAL_EPISODES = 300
ROLLOUT_EPISODES = 10
MAX_STEPS = 200

print("\n[INIT] PPO training started")
print(f"[INFO] Environment: {ENV_NAME} (continuous-control)")
print("[INFO] Purpose: algorithm–environment alignment\n")

episode_counter = 0

for update in range(TOTAL_EPISODES // ROLLOUT_EPISODES):

    batch_states = []
    batch_actions = []
    batch_log_probs = []
    batch_returns = []

    for _ in range(ROLLOUT_EPISODES):
        state, _ = env.reset()

        states = []
        actions = []
        log_probs = []
        rewards = []

        for _ in range(MAX_STEPS):
            action, log_prob = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)

            # Reward normalization for Pendulum
            reward = reward / 10.0
            done = terminated or truncated

            states.append(state)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)

            state = next_state
            if done:
                break

        # -------------------------------------------------
        # Store episode return (FOR ANALYSIS)
        # -------------------------------------------------
        episode_returns.append(sum(rewards))

        # Discounted returns (for PPO update)
        G = 0
        returns = []
        for r in reversed(rewards):
            G = r + ppo_cfg["gamma"] * G
            returns.insert(0, G)

        batch_states.extend(states)
        batch_actions.extend(actions)
        batch_log_probs.extend(log_probs)
        batch_returns.extend(returns)

        episode_counter += 1
        print(f"[TRAIN] PPO | Episode {episode_counter}/{TOTAL_EPISODES}")

    # -------------------------------------------------
    # PPO Update
    # -------------------------------------------------
    states = torch.tensor(np.array(batch_states), dtype=torch.float32).to(device)
    actions = torch.tensor(np.array(batch_actions), dtype=torch.float32).to(device)
    old_log_probs = torch.stack(batch_log_probs).detach().to(device)
    returns = torch.tensor(batch_returns, dtype=torch.float32).to(device)

    agent.learn(states, actions, old_log_probs, returns)

    print(f"[UPDATE] PPO | Batch size: {len(batch_states)}\n")

# -------------------------------------------------
# Save model & episode returns
# -------------------------------------------------
os.makedirs("checkpoints", exist_ok=True)
torch.save(agent.policy.state_dict(), "checkpoints/ppo_pendulum.pth")

os.makedirs("analysis", exist_ok=True)
np.save("analysis/ppo_episode_returns.npy", episode_returns)
print("[OK] Saved episode returns for analysis")

print("[DONE] PPO training complete")
env.close()
