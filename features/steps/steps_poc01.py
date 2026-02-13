"""
steps_taxi.py

BDD step definitions for Gymnasium Taxi-v3 environment.
Tests environment creation, reset behavior, and step interaction.
"""

from behave import given, when, then
import gymnasium as gym

# Environment Creation
# -----------------------------
@given("a Taxi-v3 environment")
def step_create_environment(context):
    """
    Initialize the Taxi-v3 environment.
    This sets up the state space, action space, and reward structure.
    """
    context.env = gym.make("Taxi-v3")

# Validate Action Space
# -----------------------------
@then("the action space size should be 6")
def step_validate_action_space(context):
    """
    Taxi-v3 has 6 discrete actions.
    0: south, 1: north, 2: east, 3: west, 4: pickup, 5: dropoff
    """
    assert context.env.action_space.n == 6

# Environment Reset
# -----------------------------
@when("the environment is reset")
def step_reset_environment(context):
    """
    Reset environment to start a new episode.
    Returns:
    - state: initial environment state (0–499)
    - info: additional info dictionary
    """
    context.state, context.info = context.env.reset()

# Validate Initial State
# -----------------------------
@then("the initial state should be between 0 and 499")
def step_validate_initial_state(context):
    """
    Taxi-v3 has 500 discrete states (0–499).
    Ensures reset produced a valid state.
    """
    assert 0 <= context.state <= 499

# Execute One Random Step
# -----------------------------
@when("a random action is taken")
def step_take_random_action(context):
    """
    Sample a random action and execute one step.
    Updates:
    - state: next state
    - reward: immediate reward for this step
    - terminated: True if task is successfully completed
    - truncated: True if episode ended due to time limit
    - info: optional diagnostic information
    """
    action = context.env.action_space.sample()
    (
        context.state,
        context.reward,
        context.terminated,
        context.truncated,
        context.info,
    ) = context.env.step(action)

# Validate Step Output
# -----------------------------
@then("the returned state should be between 0 and 499")
def step_validate_state(context):
    """
    Ensure the returned state remains within valid range (0–499).
    """
    assert 0 <= context.state <= 499


@then("the reward should be an integer")
def step_validate_reward(context):
    """
    Taxi rewards are integer values.
    """
    assert isinstance(context.reward, int)


@then("the episode status should be valid")
def step_validate_episode_status(context):
    """
    Terminated and truncated should be boolean values.
    """
    assert isinstance(context.terminated, bool)
    assert isinstance(context.truncated, bool)

# Execute Multiple Random Steps
# -----------------------------
@when("5 random actions are executed")
def step_run_multiple_steps(context):
    """
    Simulate running 5 random actions in the environment.
    Accumulate total reward for the episode
    """
    context.total_reward = 0

    for _ in range(5):
        action = context.env.action_space.sample()
        (
            state,
            reward,
            terminated,
            truncated,
            info,
        ) = context.env.step(action)

        # Add step reward to total reward
        context.total_reward += reward

        # Stop if episode ends (terminated or truncated)
        if terminated or truncated:
            break

# Validate Total Reward
# -----------------------------
@then("the total reward should be calculated")
def step_validate_total_reward(context):
    """
    Ensure total reward exists and is numeric after multiple steps.
    """
    assert isinstance(context.total_reward, int)
