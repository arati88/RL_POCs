"""
Step definitions for Environment behavior.

This file connects Gherkin scenarios from environment.feature
to the Environment implementation.
"""

from behave import given, when, then
from environment import Environment


@given("a new environment")
def step_create_environment(context):
    """
    Create a new Environment instance with a randomly generated state.
    """
    context.env = Environment()


@given("a new environment with state {value:d}")
def step_create_environment_with_state(context, value):
    """
    Create a new Environment instance and override its state.

    This allows deterministic testing instead of relying on randomness.
    """
    context.env = Environment()
    context.env.state = value  # Override random state for deterministic testing


@then("the state should be between 0 and 10")
def step_validate_state_range(context):
    """
    Validate that the generated state falls within expected bounds.
    """
    assert 0 <= context.env.state <= 10


@when("the agent takes action {action:d}")
def step_take_action(context, action):
    """
    Execute one environment step using the given action.
    Capture returned values for later assertions.
    """
    state, reward, done = context.env.step(action)
    context.returned_state = state
    context.reward = reward
    context.done = done


@then("the reward should be {expected:d}")
def step_validate_reward(context, expected):
    """
    Verify that the reward matches expected outcome.
    """
    assert context.reward == expected


@then("the episode should be done")
def step_validate_done(context):
    """
    Ensure that the environment marks the episode as completed.
    (This environment uses single-step episodes.)
    """
    assert context.done is True

