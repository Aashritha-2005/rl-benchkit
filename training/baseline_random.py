import gymnasium as gym
import numpy as np

NUM_EPISODES = 5
env = gym.make("CartPole-v1")

scores = []

print("\n🎲 Evaluating RANDOM agent\n")

for ep in range(NUM_EPISODES):
    state, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = env.action_space.sample()  # RANDOM
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward

    scores.append(total_reward)
    print(f"Episode {ep + 1}: Score = {total_reward}")

env.close()

print("\n📊 Random Agent Summary")
print(f"Average Score: {np.mean(scores):.2f}")
