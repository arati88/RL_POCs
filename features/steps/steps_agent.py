"""
Step definitions for RLAgent behavior.

This file connects Gherkin scenarios from agent.feature
to the RLAgent implementation.
"""

from behave import given, when, then
from agent import RLAgent


@given("a new RL agent")
def step_create_agent(context):
    """
    Create a fresh RLAgent instance for testing.
    """
    context.agent = RLAgent()


@then("the total reward should be 0")
def step_check_initial_reward(context):
    """
    Verify that the agent initializes with zero accumulated reward.
    """
    assert context.agent.total_reward == 0


@when("the agent selects an action for state {state:d}")
def step_select_action(context, state):
    """
    Simulate action selection for a given state.
    """
    context.action = context.agent.select_action(state)


@then("the selected action should be either 0 or 1")
def step_validate_action(context):
    """
    Ensure the selected action is within valid action space.
    """
    assert context.action in [0, 1]


@when("the agent receives reward {value:d}")
def step_update_reward(context, value):
    """
    Update agent's total reward with given value.
    """
    context.agent.update_reward(value)


@then("the total reward should be {expected:d}")
def step_check_updated_reward(context, expected):
    """
    Verify that reward accumulation behaves correctly.
    """
    assert context.agent.total_reward == expected
