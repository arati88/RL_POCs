# ============================================================
# POC-08
# Reward Curve Visualization & Evaluation
# Fraud Detection using PPO (Policy Gradient Method)
# ============================================================


# Standard library modules
import json                     # For saving experiment results
import os                       # For creating directories
from datetime import datetime   # For timestamped filenames

# Third-party numerical & ML libraries
import numpy as np              
import torch                    
import torch.nn as nn           
import torch.optim as optim     
import matplotlib.pyplot as plt 

# Dataset & evaluation utilities
from sklearn.datasets import make_classification
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

# ============================================================
# 2. DATA GENERATION (Synthetic Fraud Dataset)
# ============================================================

"""
We generate a synthetic fraud dataset using sklearn.

Why synthetic?
-------------
- Allows controlled experimentation.
- Lets us define fraud imbalance.
- Avoids privacy concerns.

Dataset characteristics:
- 6000 samples
- 20 features
- 12 informative (carry predictive signal)
- 4 redundant (correlated noise)
- Fraud rate = 8% (realistic imbalance scenario)
"""

X, y = make_classification(
    n_samples=6000,
    n_features=20,
    n_informative=12,
    n_redundant=4,
    weights=[0.92, 0.08],   # 8% fraud
    random_state=42,
)

print(f"Fraud Rate: {round(sum(y) / len(y), 3)}")

# Convert numpy arrays into PyTorch tensors
# FloatTensor for features (continuous values)
# LongTensor for class labels (required for categorical ops)
X = torch.FloatTensor(X)
y = torch.LongTensor(y)

# 80/20 train-test split
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# ============================================================
# 3. PPO POLICY NETWORK
# ============================================================

class PPOPolicy(nn.Module):
    """
    Policy Network for Fraud Classification.

    This is NOT a traditional classifier trained with cross-entropy.
    Instead, it outputs a probability distribution used by PPO.

    Architecture:
        Input (20 features)
            ↓
        Dense(64) + ReLU
            ↓
        Dense(32) + ReLU
            ↓
        Dense(2) + Softmax

    Output:
        Probability of:
            Class 0 → Legitimate
            Class 1 → Fraud
    """

    def __init__(self, input_dim: int):
        super().__init__()

        # Sequential feedforward network
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),  # First hidden layer
            nn.ReLU(),                 # Non-linearity
            nn.Linear(64, 32),         # Second hidden layer
            nn.ReLU(),
            nn.Linear(32, 2),          # Output layer (2 classes)
            nn.Softmax(dim=-1),        # Convert logits → probabilities
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Returns probability distribution over actions.
        """
        return self.network(x)


# Instantiate policy network
policy = PPOPolicy(X_train.shape[1])

# Adam optimizer for policy parameters
optimizer = optim.Adam(policy.parameters(), lr=0.0007)

# ============================================================
# 4. PPO TRAINING LOOP
# ============================================================

"""
Key PPO Concepts Used Here:
----------------------------
1. Policy sampling using Categorical distribution
2. Log probability tracking
3. Advantage normalization
4. Clipped surrogate objective
5. Multiple policy updates per batch

This is a simplified PPO variant without a critic.
"""

episodes = 300           # Number of training episodes
batch_size = 256         # Samples per episode
clip_epsilon = 0.0005       # PPO clipping threshold
reward_history = []      # Store reward progression

for episode in range(episodes):

    # --------------------------------------------------------
    # 4.1 Sample Random Mini-Batch
    # --------------------------------------------------------
    indices = np.random.choice(len(X_train), batch_size)
    batch_X = X_train[indices]
    batch_y = y_train[indices]

    # --------------------------------------------------------
    # 4.2 Forward Pass & Action Sampling
    # --------------------------------------------------------
    probs = policy(batch_X)

    # Create categorical distribution over actions
    dist = torch.distributions.Categorical(probs)

    # Sample actions (stochastic policy)
    actions = dist.sample()

    # Store old log probabilities for PPO ratio calculation
    old_log_probs = dist.log_prob(actions).detach()

    # --------------------------------------------------------
    # 4.3 Business-Aligned Reward Function
    # --------------------------------------------------------
    """
    Reward design reflects financial risk:

    True Positive  → +6
    True Negative  → +2
    False Positive → -4
    False Negative → -10  (worst case: fraud missed)

    This forces the model to prioritize catching fraud.
    """

    rewards = []

    for action, label in zip(actions.numpy(), batch_y.numpy()):

        if action == 1 and label == 1:
            reward = 6.0

        elif action == 1 and label == 0:
            reward = -4.0

        elif action == 0 and label == 1:
            reward = -10.0

        else:
            reward = 2.0

        rewards.append(reward)

    rewards = torch.FloatTensor(rewards)

    # Track mean reward per episode
    episode_reward = rewards.mean().item()
    reward_history.append(episode_reward)

    # --------------------------------------------------------
    # 4.4 Advantage Computation
    # --------------------------------------------------------
    """
    Since no critic is used, baseline = mean reward.
    We normalize advantages for stability.
    """

    advantages = rewards - rewards.mean()
    advantages = (
        (advantages - advantages.mean())
        / (advantages.std() + 1e-8)
    )

    # --------------------------------------------------------
    # 4.5 PPO Update (Clipped Objective)
    # --------------------------------------------------------
    for _ in range(4):

        new_probs = policy(batch_X)
        new_dist = torch.distributions.Categorical(new_probs)
        new_log_probs = new_dist.log_prob(actions)

        # Importance sampling ratio
        ratio = torch.exp(new_log_probs - old_log_probs)

        # Surrogate objectives
        surr1 = ratio * advantages
        surr2 = (
            torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon)
            * advantages
        )

        # PPO loss (negative because we maximize objective)
        loss = -torch.min(surr1, surr2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# ============================================================
# 5. FINAL EVALUATION
# ============================================================

"""
Evaluate trained policy deterministically.
We take argmax of probabilities.
"""

with torch.no_grad():
    probs = policy(X_test)
    preds = torch.argmax(probs, dim=1)

precision = precision_score(y_test, preds)
recall = recall_score(y_test, preds)
f1 = f1_score(y_test, preds)
accuracy = accuracy_score(y_test, preds)

print("\nFinal Classification Metrics")
print(f"Precision: {round(precision, 4)}")
print(f"Recall: {round(recall, 4)}")
print(f"F1: {round(f1, 4)}")
print(f"Accuracy: {round(accuracy, 4)}")

# ============================================================
# 6. CONVERGENCE ANALYSIS
# ============================================================

"""
We smooth reward curve using moving average.
If reward variance in last 20 episodes is small,
we consider the model stable.
"""

window = 10
moving_avg = np.convolve(
    reward_history,
    np.ones(window) / window,
    mode="valid",
)

reward_std = np.std(reward_history[-20:])
convergence_status = (
    "Stable"
    if reward_std < 5
    else "Still Learning"
)

print(f"\nConvergence Status: {convergence_status}")

# ============================================================
# 7. REWARD CURVE VISUALIZATION
# ============================================================

os.makedirs("plots", exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
plot_path = f"plots/poc08_reward_curve_{timestamp}.png"

plt.figure(figsize=(10, 6))
plt.plot(reward_history, alpha=0.6, label="Episode Reward")
plt.plot(
    range(window - 1, len(reward_history)),
    moving_avg,
    linewidth=2,
    label="Moving Average",
)

plt.title("PPO Reward Curve - Fraud Detection")
plt.xlabel("Episode")
plt.ylabel("Average Reward")
plt.legend()
plt.grid(True)

plt.savefig(plot_path)
plt.show()

print(f"Reward curve saved: {plot_path}")

# ============================================================
# 8. SAVE RESULTS
# ============================================================

results = {
    "fraud_rate": float(sum(y) / len(y)),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1),
    "accuracy": float(accuracy),
    "convergence_status": convergence_status,
    "final_avg_reward": float(np.mean(reward_history[-20:])),
    "reward_std_last_20": float(reward_std),
    "total_episodes": episodes,
}

with open("poc08_results.json", "w", encoding="utf-8") as file:
    json.dump(results, file, indent=4)

print("JSON results saved: poc08_results.json")