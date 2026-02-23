from behave import given, when, then
import pandas as pd
import os

from Policy_gradient import (
    generate_data,
    RetailEnv,
    PolicyGradientAgent,
    train,
    evaluate,
    evaluate_always_coupon,
    evaluate_never_coupon
)

@given("a synthetic retail dataset is generated")
def step_generate_dataset(context):
    if os.path.exists("retail_100_rows.csv"):
        context.df = pd.read_csv("retail_100_rows.csv")
    else:
        context.df = generate_data()


@given("a Retail Environment is created")
def step_create_env(context):
    context.env = RetailEnv(context.df)


@given("a Policy Gradient agent is initialized")
def step_initialize_agent(context):
    context.agent = PolicyGradientAgent(
        context.env.num_states,
        context.env.num_actions
    )


@when("the agent is trained for 3000 episodes")
def step_train_agent(context):
    train(context.env, context.agent, episodes=3000)


@then("the RL optimized policy should achieve higher profit than always giving coupon")
def step_compare_always(context):
    context.rl_profit = evaluate(context.env, context.agent)
    context.always_profit = evaluate_always_coupon(context.env)

    assert context.rl_profit > context.always_profit, \
        f"RL Profit {context.rl_profit} is not greater than Always Coupon {context.always_profit}"


@then("the RL optimized policy should achieve higher profit than never giving coupon")
def step_compare_never(context):
    context.never_profit = evaluate_never_coupon(context.env)

    assert context.rl_profit > context.never_profit, \
        f"RL Profit {context.rl_profit} is not greater than Never Coupon {context.never_profit}"