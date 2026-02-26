import numpy as np
import torch
import torch.nn.functional as F
from Nueral_policy_approximator import StockEnv, PolicyNetwork

# -----------------------------
# Background steps
# -----------------------------

from behave import given, when, then

@given("a fresh stock trading environment")
def step_impl(context):
    context.env = StockEnv()
    context.state = context.env.reset()

@given("a fresh policy network")
def step_impl(context):
    context.policy = PolicyNetwork()

@given("the agent has initial cash of {cash}")
def step_impl(context, cash):
    context.env = StockEnv(initial_cash=float(cash))
    context.state = context.env.reset()

@given("the agent has bought stock at price {price}")
def step_impl(context, price):
    context.env = StockEnv(initial_cash=10000)
    context.env.reset()
    # Force buy
    context.env.price = float(price)
    context.state, context.reward, _, context.info = context.env.step(1)
    assert context.env.shares > 0, "BUY did not execute properly"

# -----------------------------
# Policy network steps
# -----------------------------

@when("the policy network is queried with the current state")
def step_impl(context):
    state_tensor = torch.FloatTensor(context.state)
    context.logits = context.policy(state_tensor)

@then("it should return 3 action logits")
def step_impl(context):
    assert context.logits.shape[0] == 3, f"Expected 3 logits, got {context.logits.shape[0]}"

# -----------------------------
# BUY/SELL environment steps
# -----------------------------

@when("the agent buys at price {price}")
def step_impl(context, price):
    context.env.price = float(price)
    context.state, context.reward, _, context.info = context.env.step(1)  # BUY

@when("price increases to {price} and agent sells")
def step_impl(context, price):
    context.env.price = float(price)
    
    # Advance steps to clear cooldown
    for _ in range(context.env.cooldown):
        # HOLD action to reduce cooldown
        context.state, context.reward, _, context.info = context.env.step(0)
    
    # Now SELL
    context.state, context.reward, _, context.info = context.env.step(2)  # SELL

@then("position should be {pos}")
def step_impl(context, pos):
    expected = int(pos)
    actual = 1 if context.env.shares > 0 else 0
    assert actual == expected, f"Expected position {expected}, got {actual}"

@then("balance should decrease")
def step_impl(context):
    assert context.env.cash < context.env.initial_cash, \
        f"Expected cash < {context.env.initial_cash}, got {context.env.cash}"

@then("reward should be zero")
def step_impl(context):
    assert context.reward == 0.0, f"Expected reward 0.0, got {context.reward}"

@then("reward should be positive")
def step_impl(context):
    assert context.reward > 0, f"Expected positive reward, got {context.reward}"