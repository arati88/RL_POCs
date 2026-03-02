# PPO + Bernoulli Policy — Fraud Detection

A Reinforcement Learning based Fraud Detection Engine built using  
**Proximal Policy Optimization (PPO)** with a **Bernoulli stochastic policy**.

This POC simulates a real-time fraud detection system that learns to:

- Block fraudulent transactions
- Minimize customer friction
- Optimize long-term business reward
- Maintain policy stability using PPO clipping

---

# Overview

Traditional fraud detection uses supervised learning.

This project uses **Reinforcement Learning** to:

- Optimize asymmetric fraud costs
- Adapt to evolving fraud patterns
- Learn from reward signals instead of labels alone
- Maintain stable policy updates via PPO

---

# Architecture

State (5 features per transaction):
- Amount (log scaled)
- Time of day
- Transaction velocity
- Geo risk
- Device trust

Action Space (Bernoulli Policy):
- 0 → Allow
- 1 → Block

Policy:
- 2-layer MLP → logits → sigmoid → Bernoulli distribution

Value Function:
- 2-layer MLP → V(s)

Algorithm:
- PPO with clipped surrogate objective
- Clipped value function loss
- Entropy bonus
- GAE (γ=0.99, λ=0.95)

Optimizer:
- Adam (separate learning rates for actor & critic)
- Exponential learning rate decay
- Gradient clipping

---

# Reward Design

| Outcome | Reward |
|----------|--------|
| True Positive (Block fraud) | +2.0 |
| True Negative (Allow legit) | +0.5 |
| False Negative (Miss fraud) | -2.0 |
| False Positive (Block legit) | -0.8 |

The reward function reflects real-world business priorities:
- Missing fraud is most costly
- Customer friction is penalized
- Correct fraud blocking is strongly rewarded

---
# Training
- Train PPO agent
- Tune threshold for best F1
- Run inference demo

# Metrics Tracked

During training:

- Average Reward
- Accuracy
- Precision
- Recall
- F1 Score
- Entropy
- KL Divergence
- Actor Loss

# Threshold Optimization

Post-training, threshold tuning is performed to:
Maximize F1 score on held-out data.
This mimics real-world fraud system calibration.

# BDD Testing (Behave)

This project includes Gherkin-based tests to validate:

- PPO training stability
- Fraud probability separation
- Threshold tuning improvement
- Recall requirements
- KL divergence stability

Run tests:
    python -m behave

Expected output:
    1 feature passed
    5 scenarios passed
    23 steps passed
