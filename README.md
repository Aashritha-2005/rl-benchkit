# RLBenchKit

A lightweight framework for benchmarking reinforcement learning algorithms
across discrete and continuous control tasks.

---

## Overview

RLBenchKit is an **engineering-focused reinforcement learning framework** designed to
train, evaluate, and reason about **value-based and policy-based agents** under
controlled experimental settings.

Unlike tutorial-style repositories, this project emphasizes:

- algorithm correctness
- learning stability
- failure modes
- reproducible experimentation
- **algorithm–environment alignment**

The goal is not leaderboard performance, but **clean, defensible reinforcement learning systems**.

---

## Algorithms Implemented

- **Double DQN**
- **Dueling DQN**
- **PPO (Policy Gradient)**

All algorithms are implemented from scratch using a **unified agent abstraction**.

---

## Environments

- **CartPole-v1** — discrete control (DQN)
- **Pendulum-v1** — continuous control (PPO)

Environments are chosen deliberately to match algorithm assumptions rather than for score chasing.

---

## Key Contributions

- Unified abstraction for value-based and policy-gradient agents
- Deterministic evaluation and reproducible training loops
- Explicit handling of learning stability and failure modes
- Clean separation between discrete and continuous control settings
- Minimal, extensible engineering structure

---

## Experimental Focus

This project was designed to demonstrate:

- Correct implementation of deep reinforcement learning algorithms
- Awareness of **algorithm–environment suitability**
- Stable training behavior under realistic constraints
- Engineering restraint and reproducibility
- Identification of common RL failure modes such as instability, variance sensitivity, and sample inefficiency


Rather than maximizing scores, the emphasis is on:

- correctness
- stability
- transparent evaluation
- honest analysis

---

## Results Summary

- **DQN** demonstrates fast and stable learning on **CartPole-v1**, a low-dimensional discrete control task.
- **PPO** demonstrates stable learning behavior on **Pendulum-v1**, a continuous-control environment.

**PPO was additionally evaluated on a continuous-control task (Pendulum-v1), demonstrating correct algorithm–environment selection rather than score maximization.**

Direct numerical comparison across environments is intentionally avoided.

---

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Train DQN on CartPole:

```bash
python -m training.train_dqn
```

Train PPO on Pendulum:

```bash
python -m training.train_ppo
```

Generate learning-curve visualization:

```bash
python -m analysis.plots
```
---

## RLBenchKit prioritizes:

- clarity over complexity

- correctness over benchmarks

- understanding over optimization
# rl-benchkit
A reproducible benchmarking framework for evaluating reinforcement learning algorithms across discrete and continuous control tasks.
