"""
BDD Step Definitions for POC-02: Q-Learning in Taxi-v3 Environment

This file connects Gherkin feature steps to actual Python test logic.

We validate:
1. Q-table initialization
2. Q-value updates (Bellman equation correctness)
3. Epsilon decay behavior
4. Learning performance improvement
"""

# ============================================================
# Imports
# ============================================================

from behave import given, when, then   # BDD decorators
import gymnasium as gym               # RL environment
import numpy as np                    # Numerical operations


# ============================================================
# Global Shared Variables (Test State)
# ============================================================

# Environment instance (shared across steps)
env = None

# Q-table (State × Action value table)
q_table = None

# Learning parameters (used in update tests)
alpha = 0.1   # Learning rate
gamma = 0.9   # Discount factor
epsilon = 1.0 # Exploration rate


# ============================================================
# Scenario 1: Q-table should initialize correctly
# ============================================================

@given("the Taxi environment is created")
def step_create_env(context):
    """
    Create Taxi-v3 environment.

    Taxi-v3:
    - 500 discrete states
    - 6 discrete actions
    """
    global env
    env = gym.make("Taxi-v3")


@when("the Q-table is initialized")
def step_init_qtable(context):
    """
    Initialize Q-table with zeros.

    Q-table shape:
        rows    → number of states
        columns → number of actions
    """
    global q_table
    q_table = np.zeros((env.observation_space.n,
                        env.action_space.n))


@then("the Q-table should have 500 rows and 6 columns")
def step_check_shape(context):
    """
    Validate Q-table dimensions.

    Expected shape for Taxi-v3:
        (500 states, 6 actions)
    """
    assert q_table.shape == (500, 6)


# ============================================================
# Scenario 2: Q-values should update after one learning step
# ============================================================

@given("a Q-table initialized with zeros")
def step_zero_qtable(context):
    """
    Create a fresh Taxi environment
    and initialize Q-table to all zeros.
    """
    global q_table, env
    env = gym.make("Taxi-v3")
    q_table = np.zeros((env.observation_space.n,
                        env.action_space.n))


@when("a learning update is performed")
def step_learning_update(context):
    """
    Perform ONE Q-learning update.

    Bellman Update Rule:

    Q(s,a) = Q(s,a) + α * (r + γ * max(Q(s')) - Q(s,a))

    Where:
        s  → current state
        a  → action taken
        r  → reward received
        s' → next state
    """

    # Reset environment to get initial state
    state, info = env.reset()

    # Choose random action
    action = env.action_space.sample()

    # Execute action
    next_state, reward, terminated, truncated, info = env.step(action)

    # Store original Q-value for comparison
    old_value = q_table[state, action]

    # Apply Bellman update equation
    q_table[state, action] = old_value + alpha * (
        reward
        + gamma * np.max(q_table[next_state])
        - old_value
    )

    # Store values for assertion step
    context.updated_value = q_table[state, action]
    context.old_value = old_value


@then("the Q-value for the selected state-action should change")
def step_check_update(context):
    """
    Ensure that Q-value has been updated.

    Since the Q-table started at zeros,
    a valid update should modify the value.
    """
    assert context.updated_value != context.old_value


# ============================================================
# Scenario 3: Epsilon should decay after an episode
# ============================================================

@given("epsilon is initialized to 1.0")
def step_epsilon_init(context):
    """
    Initialize exploration rate (epsilon)
    to 1.0 (100% exploration).
    """
    context.epsilon = 1.0


@when("epsilon decay is applied")
def step_decay(context):
    """
    Apply epsilon decay strategy.

    Formula:
        epsilon = epsilon * decay_rate
    """
    epsilon_decay = 0.995
    context.epsilon = context.epsilon * epsilon_decay


@then("epsilon should be less than 1.0")
def step_check_decay(context):
    """
    Validate that epsilon decreased
    after applying decay.
    """
    assert context.epsilon < 1.0


# ============================================================
# Scenario 4: Agent reward should improve after training
# ============================================================

@given("the agent is trained for multiple episodes")
def step_train_agent(context):
    """
    Train Q-learning agent for multiple episodes
    and compute average reward of last 100 episodes.

    This validates that learning actually occurs.
    """

    env = gym.make("Taxi-v3")

    env.reset(seed=42)
    np.random.seed(42)

    # Initialize Q-table
    q_table = np.zeros((env.observation_space.n,
                        env.action_space.n))

    # Training hyperparameters
    alpha = 0.1
    gamma = 0.9
    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.01

    rewards = []

    # Train for 1000 episodes
    for episode in range(1000):

        state, info = env.reset()
        done = False
        total_reward = 0

        while not done:

            # Epsilon-greedy action selection
            if np.random.random() < epsilon:
                action = env.action_space.sample()  # Explore
            else:
                action = np.argmax(q_table[state])  # Exploit

            # Take action in environment
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Apply Q-learning update
            q_table[state, action] += alpha * (
                reward
                + gamma * np.max(q_table[next_state])
                - q_table[state, action]
            )

            state = next_state
            total_reward += reward

        # Decay exploration rate
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        rewards.append(total_reward)

    # Compute evaluation metric
    context.avg_reward = np.mean(rewards[-100:])

    env.close()


@then("the average reward of the last 100 episodes should show learning improvement")
def step_check_reward(context):
    """
    Random baseline ≈ -200
    Trained agent should improve significantly.

    We assert reward better than -20 to confirm learning.
    """
    print("Average reward:", context.avg_reward)
    assert context.avg_reward > -20
