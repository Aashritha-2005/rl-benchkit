import os
import numpy as np
import matplotlib.pyplot as plt


def plot(values, title, save_path=None):
    """
    Plot episode returns or rolling averages.

    Args:
        values (list or np.ndarray): values to plot
        title (str): plot title
        save_path (str, optional): path to save plot (PNG)
    """
    plt.figure(figsize=(8, 4))
    plt.plot(values)
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel("Return")

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
        print(f"[OK] Plot saved to {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    """
    Generates a learning-curve plot for PPO on Pendulum-v1.
    Expects episode returns saved from training.
    """

    returns_path = "analysis/ppo_episode_returns.npy"

    if not os.path.exists(returns_path):
        print("[INFO] No episode returns found. Run PPO training first.")
        exit(0)

    returns = np.load(returns_path)

    # Rolling average (lightweight analysis)
    window = 50
    if len(returns) >= window:
        rolling_avg = np.convolve(
            returns, np.ones(window) / window, mode="valid"
        )
        plot(
            values=rolling_avg,
            title="PPO on Pendulum-v1 (Rolling Avg Return)",
            save_path="analysis/ppo_pendulum_learning_curve.png"
        )
    else:
        plot(
            values=returns,
            title="PPO on Pendulum-v1 (Episode Returns)",
            save_path="analysis/ppo_pendulum_learning_curve.png"
        )
