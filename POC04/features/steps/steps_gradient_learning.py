import numpy as np
from behave import given, when, then
from policy_gradient import (
    RetailEnv,
    PolicyGradientAgent,
    train,
    evaluate,
    evaluate_always_coupon,
    evaluate_never_coupon,
    NORMAL_BUYER,
    NO_COUPON,
)


# ============================================================
# GIVEN
# ============================================================

@given("a retail environment")
def step_given_env(context):
    context.env = RetailEnv(rng_seed=42)


@given("a policy gradient agent")
def step_given_agent(context):
    context.agent = PolicyGradientAgent(
        context.env.num_states,
        context.env.num_actions,
    )


@given("a trained policy gradient agent")
def step_given_trained_agent(context):
    context.env = RetailEnv(rng_seed=42)
    context.agent = PolicyGradientAgent(
        context.env.num_states,
        context.env.num_actions,
    )
    train(context.env, context.agent, episodes=1500)


# ============================================================
# WHEN
# ============================================================

@when("I take action GiveCoupon")
def step_when_take_action(context):
    context.env.reset()
    next_state, reward, done = context.env.step(1)
    context.next_state = next_state
    context.reward = reward


@when("I train the agent for 100 episodes")
def step_when_train(context):
    train(context.env, context.agent, episodes=100)


@when("I evaluate all strategies")
def step_when_evaluate(context):
    context.rl_profit = evaluate(context.env, context.agent)
    context.always_profit = evaluate_always_coupon(context.env)
    context.never_profit = evaluate_never_coupon(context.env)


@when("I inspect the learned policy")
def step_when_inspect_policy(context):
    probs = context.agent.get_action_probs(NORMAL_BUYER)
    context.normal_buyer_action = int(np.argmax(probs))


# ============================================================
# THEN
# ============================================================

@then("the environment should have 3 states")
def step_then_state_count(context):
    assert context.env.num_states == 3


@then("the environment should have 2 actions")
def step_then_action_count(context):
    assert context.env.num_actions == 2


@then("the next state should be valid")
def step_then_valid_state(context):
    assert context.next_state in [0, 1, 2]


@then("the reward should be a float")
def step_then_reward_float(context):
    assert isinstance(context.reward, float)


@then("training should complete successfully")
def step_then_training_complete(context):
    assert context.agent.theta is not None


@then("the RL policy profit should be greater than always coupon")
def step_then_rl_beats_always(context):
    assert context.rl_profit > context.always_profit


@then("the RL policy profit should be greater than never coupon")
def step_then_rl_beats_never(context):
    assert context.rl_profit > context.never_profit


@then("NormalBuyer should prefer NoCoupon")
def step_then_normal_buyer_logic(context):
    assert context.normal_buyer_action == NO_COUPON