"""
steps_gradient_learning.py

BDD step definitions for POC-04: Retail Coupon Optimization via Policy Gradient.
Tests environment setup, reward correctness, transition validity, and RL performance.
"""

import sys
import os

# Allow imports from POC04 directory when behave is run from there
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from behave import given, when, then
import numpy as np

from Policy_gradient import (
    RetailEnv,
    PolicyGradientAgent,
    train,
    evaluate,
    evaluate_always_coupon,
    evaluate_never_coupon,
    NEUTRAL,
    NO_COUPON,
    GIVE_COUPON,
    DEFAULT_MARGIN,
    DEFAULT_COUPON_COST,
)


# ============================================================
# ENVIRONMENT SETUP STEPS
# ============================================================

@given("a Retail Environment is created with default config")
def step_create_env_default(context):
    """Create a RetailEnv using default configuration."""
    context.env = RetailEnv(rng_seed=42)


@given("a Retail Environment is created with seed {seed:d} and always-no-purchase probabilities")
def step_create_env_no_purchase(context, seed):
    """
    Create a RetailEnv where purchase never occurs (p=0.0 for all state-action pairs).
    Used to deterministically verify reward=0 when no purchase happens.
    """
    from Policy_gradient import NUM_STATES, NUM_ACTIONS, DEFAULT_TRANSITION_PROBS
    no_buy_probs = {
        (s, a): 0.0
        for s in range(NUM_STATES)
        for a in range(NUM_ACTIONS)
    }
    context.env = RetailEnv(
        purchase_probs=no_buy_probs,
        transition_probs=DEFAULT_TRANSITION_PROBS,
        rng_seed=seed,
    )
    context.env.state = NEUTRAL


@given("a Retail Environment is created with seed {seed:d} and always-purchase probabilities")
def step_create_env_always_purchase(context, seed):
    """
    Create a RetailEnv where purchase always occurs (p=1.0 for all state-action pairs).
    Used to deterministically verify profit reward calculation.
    """
    from Policy_gradient import NUM_STATES, NUM_ACTIONS, DEFAULT_TRANSITION_PROBS
    always_buy_probs = {
        (s, a): 1.0
        for s in range(NUM_STATES)
        for a in range(NUM_ACTIONS)
    }
    context.env = RetailEnv(
        purchase_probs=always_buy_probs,
        transition_probs=DEFAULT_TRANSITION_PROBS,
        rng_seed=seed,
    )
    context.env.state = NEUTRAL


@given("a Policy Gradient agent is initialized")
def step_initialize_agent(context):
    context.agent = PolicyGradientAgent(
        context.env.num_states,
        context.env.num_actions,
    )


# ============================================================
# ACTION STEPS
# ============================================================

@when("a NoCoupon action is taken from Neutral state")
def step_take_no_coupon(context):
    context.env.state = NEUTRAL
    context.next_state, context.reward, _ = context.env.step(NO_COUPON)


@when("a GiveCoupon action is taken from Neutral state")
def step_take_give_coupon(context):
    context.env.state = NEUTRAL
    context.next_state, context.reward, _ = context.env.step(GIVE_COUPON)


@when("{n:d} random steps are executed")
def step_run_random_steps(context, n):
    """Run n random steps and collect all visited states."""
    context.env.reset()
    context.visited_states = []
    for _ in range(n):
        action = int(context.env._rng.integers(0, context.env.num_actions))
        next_state, _, _ = context.env.step(action)
        context.visited_states.append(next_state)


@when("the agent is trained for 3000 episodes")
def step_train_agent(context):
    train(context.env, context.agent, episodes=3000)


# ============================================================
# ASSERTION STEPS
# ============================================================

@then("the state space size should be 3")
def step_check_state_space(context):
    """Validate that the environment exposes exactly 3 customer states."""
    assert context.env.num_states == 3, (
        f"Expected 3 states, got {context.env.num_states}"
    )


@then("the action space size should be 2")
def step_check_action_space(context):
    """Validate that the environment exposes exactly 2 actions."""
    assert context.env.num_actions == 2, (
        f"Expected 2 actions, got {context.env.num_actions}"
    )


@then("all visited states should be within the valid state space")
def step_check_transitions(context):
    """Ensure every transition stays within {0, 1, 2}."""
    invalid = [s for s in context.visited_states if s not in (0, 1, 2)]
    assert not invalid, (
        f"Transitions produced out-of-range states: {set(invalid)}"
    )


@then("the reward should be 0.0")
def step_check_zero_reward(context):
    """No purchase → reward must be 0."""
    assert context.reward == 0.0, (
        f"Expected reward 0.0 (no purchase), got {context.reward}"
    )


@then("the reward should equal the configured margin")
def step_check_margin_reward(context):
    """Purchase without coupon → reward equals margin."""
    assert context.reward == DEFAULT_MARGIN, (
        f"Expected reward {DEFAULT_MARGIN} (margin), got {context.reward}"
    )


@then("the reward should equal margin minus coupon cost")
def step_check_coupon_reward(context):
    """Purchase with coupon → reward equals margin - coupon_cost."""
    expected = DEFAULT_MARGIN - DEFAULT_COUPON_COST
    assert context.reward == expected, (
        f"Expected reward {expected} (margin - coupon_cost), got {context.reward}"
    )


@then("the RL optimized policy should achieve higher profit than always giving coupon")
def step_compare_always(context):
    context.rl_profit = evaluate(context.env, context.agent)
    context.always_profit = evaluate_always_coupon(context.env)
    assert context.rl_profit > context.always_profit, (
        f"RL profit {context.rl_profit:.3f} is not greater than "
        f"Always Coupon profit {context.always_profit:.3f}"
    )


@then("the RL optimized policy should achieve higher profit than never giving coupon")
def step_compare_never(context):
    context.never_profit = evaluate_never_coupon(context.env)
    assert context.rl_profit > context.never_profit, (
        f"RL profit {context.rl_profit:.3f} is not greater than "
        f"Never Coupon profit {context.never_profit:.3f}"
    )
