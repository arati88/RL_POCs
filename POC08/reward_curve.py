# ============================================================
# POC-08
# Reward Curve Visualization & Evaluation
# Fraud Detection using PPO
# ============================================================

import os
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.datasets import make_classification
from datetime import datetime

# ============================================================
# 1️⃣ Generate Realistic Fraud Dataset
# ============================================================

X, y = make_classification(
    n_samples=6000,
    n_features=20,
    n_informative=12,
    n_redundant=4,
    weights=[0.92, 0.08],  # 8% fraud
    random_state=42
)

print(f"Fraud Rate: {round(sum(y)/len(y),3)}")

X = torch.FloatTensor(X)
y = torch.LongTensor(y)

# Train/Test split
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# ============================================================
# 2️⃣ PPO Policy Network
# ============================================================

class PPOPolicy(nn.Module):
    def __init__(self, input_dim):
        super(PPOPolicy, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.network(x)

policy = PPOPolicy(X_train.shape[1])
optimizer = optim.Adam(policy.parameters(), lr=0.0007)

# ============================================================
# 3️⃣ PPO Training
# ============================================================

episodes = 150
batch_size = 128
clip_epsilon = 0.2
reward_history = []

for episode in range(episodes):

    indices = np.random.choice(len(X_train), batch_size)
    batch_X = X_train[indices]
    batch_y = y_train[indices]

    probs = policy(batch_X)
    dist = torch.distributions.Categorical(probs)
    actions = dist.sample()
    old_log_probs = dist.log_prob(actions).detach()

    # --------------------------------------------------------
    # Business-Aligned Reward Function
    # --------------------------------------------------------

    rewards = []

    for action, label in zip(actions.numpy(), batch_y.numpy()):

        if action == 1 and label == 1:
            reward = 6.0      # True Positive (fraud caught)

        elif action == 1 and label == 0:
            reward = -4.0     # False Positive

        elif action == 0 and label == 1:
            reward = -10.0    # False Negative (high cost)

        else:
            reward = 2.0      # True Negative

        rewards.append(reward)

    rewards = torch.FloatTensor(rewards)

    # Store average episode reward
    episode_reward = rewards.mean().item()
    reward_history.append(episode_reward)

    # --------------------------------------------------------
    # PPO Update
    # --------------------------------------------------------

    advantages = rewards - rewards.mean()
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    for _ in range(4):
        new_probs = policy(batch_X)
        new_dist = torch.distributions.Categorical(new_probs)
        new_log_probs = new_dist.log_prob(actions)

        ratio = torch.exp(new_log_probs - old_log_probs)

        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages

        loss = -torch.min(surr1, surr2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# ============================================================
# 4️⃣ Final Evaluation
# ============================================================

with torch.no_grad():
    probs = policy(X_test)
    preds = torch.argmax(probs, dim=1)

precision = precision_score(y_test, preds)
recall = recall_score(y_test, preds)
f1 = f1_score(y_test, preds)
accuracy = accuracy_score(y_test, preds)

print("\nFinal Classification Metrics")
print(f"Precision: {round(precision,4)}")
print(f"Recall: {round(recall,4)}")
print(f"F1: {round(f1,4)}")
print(f"Accuracy: {round(accuracy,4)}")

# ============================================================
# 5️⃣ Convergence Analysis
# ============================================================

window = 10
moving_avg = np.convolve(reward_history, np.ones(window)/window, mode='valid')

reward_std = np.std(reward_history[-20:])
convergence_status = "Stable" if reward_std < 5 else "Still Learning"

print(f"\nConvergence Status: {convergence_status}")

# ============================================================
# 6️⃣ Reward Curve Visualization
# ============================================================

os.makedirs("plots", exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
plot_path = f"plots/poc08_reward_curve_{timestamp}.png"

plt.figure(figsize=(10,6))
plt.plot(reward_history, alpha=0.6, label="Episode Reward")
plt.plot(range(window-1, len(reward_history)), moving_avg,
         linewidth=2, label="Moving Average")

plt.title("PPO Reward Curve - Fraud Detection")
plt.xlabel("Episode")
plt.ylabel("Average Reward")
plt.legend()
plt.grid(True)

plt.savefig(plot_path)
plt.show()

print(f"Reward curve saved: {plot_path}")

# ============================================================
# 7️⃣ Save JSON Results (For Gherkin Testing)
# ============================================================

results = {
    "fraud_rate": float(sum(y)/len(y)),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1),
    "accuracy": float(accuracy),
    "convergence_status": convergence_status,
    "final_avg_reward": float(np.mean(reward_history[-20:])),
    "reward_std_last_20": float(reward_std),
    "total_episodes": episodes
}

with open("poc08_results.json", "w") as f:
    json.dump(results, f, indent=4)

print("JSON results saved: poc08_results.json")