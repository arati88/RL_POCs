import gymnasium as gym
import numpy as np

# Create Taxi environment
# Taxi-v3:
# - 500 possible states
# - 6 discrete actions
# - Reward structure:
#     +20 for successful drop-off
#     -1 for each step
#     -10 for illegal pickup/dropoff
env = gym.make("Taxi-v3")

# Initialize Q-table (500 states × 6 actions)
# Q-table dimensions:
# Rows    → States (0 to 499)
# Columns → Actions (0 to 5)
#
# Each cell Q[s, a] represents:
# "Expected future reward if we take action 'a' in state 's'"

q_table = np.zeros((env.observation_space.n, env.action_space.n))

# Hyperparameters
alpha = 0.1       # Learning rate (how much new info overrides old info)
gamma = 0.9       # Discount factor (importance of future rewards)
epsilon = 1.0     # Exploration rate (start fully exploring)
epsilon_decay = 0.995 # Gradually reduce exploration
epsilon_min = 0.01  # Minimum exploration threshold

episodes = 5000    # Total training episodes

# Track rewards per episode to evaluate learning performance
rewards_per_episode = []

#Training Loop (Trial-and-Error Learning)
for episode in range(episodes):

    # Reset environment at start of each episode
    state, info = env.reset()
    done = False
    total_reward = 0

    # Continue until episode finishes
    while not done:

        # ε-greedy Action Selection
        # With probability epsilon → Explore (random action)
        # Otherwise → Exploit (choose best known action)
        if np.random.random() < epsilon:
            action = env.action_space.sample()   # Exploration
        else:
            action = np.argmax(q_table[state])  # Exploitation

        # Take action in environment
        next_state, reward, terminated, truncated, info = env.step(action)

        # Episode ends if task completed or time limit reached
        done = terminated or truncated

        # Q-Learning Update (Bellman Equation)
        # --------------------------------------------------
        # Q(s,a) = Q(s,a) + α [ r + γ max Q(s',a') - Q(s,a) ]
        #
        # Where:
        #   s  = current state
        #   a  = action taken
        #   r  = reward received
        #   s' = next state

        q_table[state, action] = q_table[state, action] + alpha * (
            reward + gamma * np.max(q_table[next_state]) - q_table[state, action]
        )

        # Move to next state
        state = next_state

        # Accumulate reward for this episode
        total_reward += reward

    # Decay exploration rate
    # Gradually shift from exploration to exploitation
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

     # Store total reward for analysis
    rewards_per_episode.append(total_reward)

# Close environment after training completes
env.close()

#Evaluate Learning Performance

print("Training completed.")
print("Average reward over last 100 episodes:",
      np.mean(rewards_per_episode[-100:]))
