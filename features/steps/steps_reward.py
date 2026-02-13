"""
Step definitions for testing the reward calculation logic.

This file validates the behavior of the calculate_reward function.
"""

from behave import when, then
from reward import calculate_reward


@when("reward is calculated for success {status}")
def step_calculate_reward(context, status):
    """
    Convert the Gherkin string input into a boolean value
    and calculate the reward accordingly.

    Gherkin passes parameters as strings, so explicit conversion
    to boolean is required.
    """
    is_success = True if status == "True" else False
    context.reward = calculate_reward(is_success)


@then("the reward value should be {expected:d}")
def step_validate_reward(context, expected):
    """
    Validate that the reward function returns
    the expected numeric value.
    """
    assert context.reward == expected
