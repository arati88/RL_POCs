"""
steps_taxi.py

BDD step definitions for Gymnasium Taxi-v3 environment.
Tests environment creation, reset behavior, and step interaction.
"""

from behave import given, when, then
import gymnasium as gym


@given("a Taxi-v3 environment")
def step_create_environment(context):
    """
    Initialize the Taxi-v3 environment.
    """
    context.env = gym.make("Taxi-v3")


@then("the action space size should be 6")
def step_validate_action_space(context):
    """
    Taxi-v3 has 6 discrete actions.
    """
    assert context.env.action_space.n == 6


@when("the environment is reset")
def step_reset_environment(context):
    """
    Reset environment to start a new episode.
    """
    context.state, context.info = context.env.reset()


@then("the initial state should be between 0 and 499")
def step_validate_initial_state(context):
    """
    Taxi-v3 has 500 discrete states (0–499).
    """
    assert 0 <= context.state <= 499


@when("a random action is taken")
def step_take_random_action(context):
    """
    Sample a random action and execute one step.
    """
    action = context.env.action_space.sample()
    (
        context.state,
        context.reward,
        context.terminated,
        context.truncated,
        context.info,
    ) = context.env.step(action)


@then("the returned state should be between 0 and 499")
def step_validate_state(context):
    """
    Validate state remains within valid range.
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


@when("5 random actions are executed")
def step_run_multiple_steps(context):
    """
    Execute 5 random actions and accumulate total reward.
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

        context.total_reward += reward

        if terminated or truncated:
            break


@then("the total reward should be calculated")
def step_validate_total_reward(context):
    """
    Ensure total reward exists and is numeric.
    """
    assert isinstance(context.total_reward, int)
