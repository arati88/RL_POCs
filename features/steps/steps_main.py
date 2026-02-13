"""
Step definitions for running a full one-step episode
combining RLAgent and Environment.

This file validates the interaction between agent and environment.
"""

from behave import given, when, then
from agent import RLAgent
from environment import Environment


@given("an environment with state {value:d}")
def step_create_env_with_state(context, value):
    """
    Create an environment with a deterministic state value.
    Overriding randomness ensures predictable integration testing.
    """
    context.env = Environment()
    context.env.state = value


@when("the agent runs one episode")
def step_run_episode(context):
    """
    Execute a single decision loop:

    1. Agent observes state
    2. Agent selects action
    3. Environment returns reward
    4. Agent updates total reward
    """
    state = context.env.state
    action = context.agent.select_action(state)
    _, reward, done = context.env.step(action)
    context.agent.update_reward(reward)

    # Store results for validation in subsequent steps
    context.result = {
        "state": state,
        "action": action,
        "reward": reward,
        "total_reward": context.agent.total_reward,
    }


@then("the returned state should be {expected:d}")
def step_validate_state(context, expected):
    """
    Ensure the state used during the episode matches expected value.
    """
    assert context.result["state"] == expected


@then("the action should be either 0 or 1")
def step_validate_action(context):
    """
    Validate that the agent selected a valid action
    within the defined action space.
    """
    assert context.result["action"] in [0, 1]


@then("the reward should be either 1 or -1")
def step_validate_reward(context):
    """
    Ensure reward follows the environment's binary reward logic.
    """
    assert context.result["reward"] in [1, -1]


@then("the total reward should equal the reward")
def step_validate_total_reward(context):
    """
    Since this is a single-step episode,
    total accumulated reward should match the received reward.
    """
    assert context.result["total_reward"] == context.result["reward"]

