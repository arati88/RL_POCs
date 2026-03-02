import gymnasium as gym
import numpy as np


# Create Taxi environment
# Taxi-v3:
# - 500 possible states
# - 6 discrete actions
# Reward structure:
#   +20 for successful drop-off
#   -1 for each step
#   -10 for illegal pickup/dropoff
env = gym.make("Taxi-v3")


# Initialize Q-table (500 states × 6 actions)
# Rows    → States (0 to 499)
# Columns → Actions (0 to 5)
# Each cell Q[s, a] represents:
# "Expected future reward if we take action 'a' in state 's'"
q_table = np.zeros(
    (env.observation_space.n, env.action_space.n)
)


# Hyperparameters
ALPHA = 0.1                # Learning rate
GAMMA = 0.9                # Discount factor
EPSILON = 1.0              # Initial exploration rate
EPSILON_DECAY = 0.995      # Exploration decay rate
EPSILON_MIN = 0.01         # Minimum exploration threshold
EPISODES = 5000            # Total training episodes


# Track rewards per episode
rewards_per_episode = []


# ============================
# Training Loop
# ============================
for episode in range(EPISODES):

    state, info = env.reset()
    done = False
    total_reward = 0

    while not done:

        # ε-greedy action selection
        if np.random.random() < EPSILON:
            action = env.action_space.sample()      # Explore
        else:
            action = np.argmax(q_table[state])      # Exploit

        # Take action
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Q-Learning Update (Bellman Equation)
        q_table[state, action] += ALPHA * (
            reward
            + GAMMA * np.max(q_table[next_state])
            - q_table[state, action]
        )

        state = next_state
        total_reward += reward

    # Decay exploration rate
    EPSILON = max(EPSILON_MIN, EPSILON * EPSILON_DECAY)

    rewards_per_episode.append(total_reward)


env.close()


# ============================
# Evaluation
# ============================
print("Training completed.")
print(
    "Average reward over last 100 episodes:",
    np.mean(rewards_per_episode[-100:])
)