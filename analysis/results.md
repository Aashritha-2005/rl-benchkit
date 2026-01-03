# Reinforcement Learning Results

## Experimental Setup

- Framework: Custom RL framework (PyTorch, Gymnasium)
- Hardware: CPU
- Evaluation: Deterministic evaluation for trained policies
- Goal: Validate correct algorithm–environment alignment rather than score maximization

---

## Discrete Control: CartPole-v1 (DQN)

### Setup
- Environment: CartPole-v1
- Algorithm: DQN (value-based)
- Episodes: 50
- Evaluation policy: ε = 0 (greedy)

### Results

| Metric | Observation |
|------|------------|
| Learning Stability | High |
| Convergence Speed | Fast |
| Control Performance | Stable |
| Sample Efficiency | High |

**Observation:**  
DQN converges rapidly and maintains stable pole control, consistent with expectations for a low-dimensional, deterministic control task.

---

## Continuous Control: Pendulum-v1 (PPO)

### Setup
- Environment: Pendulum-v1
- Algorithm: PPO (policy-gradient)
- Episodes: 600
- Action Space: Continuous

### Results

| Metric | Observation |
|------|------------|
| Learning Stability | Stable |
| Training Behavior | Gradual improvement |
| Policy Collapse | Not observed |
| Optimization | Smooth |

**Observation:**  
PPO exhibits stable learning behavior on a continuous-control task, with episode returns improving steadily over training.

---

## Key Insight (Critical)

The experiments demonstrate **correct algorithm–environment alignment**:

- **DQN** is well-suited for discrete, low-dimensional control problems such as CartPole.
- **PPO** is appropriate for continuous action spaces, as demonstrated on Pendulum-v1.

Direct numerical comparison across these environments is **intentionally avoided**, as the tasks differ fundamentally in dynamics and action spaces.

---

## Conclusion

This project emphasizes **correctness, stability, and reproducibility** rather than leaderboard performance.

The results validate that:
- Value-based methods are effective for discrete control.
- Policy-gradient methods are required for continuous control.

These findings are theoretically consistent and expected.
