import numpy as np

from behave import given, when, then

from PPO_Bernoulli import PPOConfig, PPOAgent, FraudEnvironment, train, tune_threshold


@given("a PPO fraud detection configuration")
def step_config(context):
    context.cfg = PPOConfig(
        total_episodes=10,
        batch_size=256,
        hidden=32,
        seed=123
    )


@given("a fraud environment")
def step_env(context):
    context.env = FraudEnvironment(context.cfg)


@when("the PPO agent is trained for 5 episodes")
def step_train_small(context):
    context.cfg.total_episodes = 5
    context.agent, context.tracker, context.threshold = train(context.cfg)


@then("the training should complete successfully")
def step_check_training(context):
    assert context.agent is not None
    assert len(context.tracker.history) > 0


# -----------------------------------------------------
# Trained Agent Fixture
# -----------------------------------------------------

@given("the trained PPO agent")
def step_trained_agent(context):
    context.agent, context.tracker, context.threshold = train(context.cfg)


@when("I evaluate 200 transactions")
def step_eval_200(context):
    txs = context.env.batch(200)
    states = np.stack([tx.state for tx in txs])
    labels = np.array([int(tx.is_fraud) for tx in txs])
    preds, probs = context.agent.predict_batch(states)

    context.labels = labels
    context.preds = preds
    context.probs = probs


@then("average fraud probability for fraud cases should be higher than legit cases")
def step_prob_check(context):
    fraud_probs = context.probs[context.labels == 1]
    legit_probs = context.probs[context.labels == 0]

    assert fraud_probs.mean() > legit_probs.mean()


# -----------------------------------------------------
# Threshold tuning
# -----------------------------------------------------

@when("I tune the decision threshold")
def step_tune_threshold(context):
    context.best_thr, context.best_f1 = tune_threshold(
        context.agent,
        context.env,
        n=1000
    )


@then("the best F1 score should be greater than 0.5")
def step_check_f1(context):
    assert context.best_f1 > 0.5


# -----------------------------------------------------
# Recall check
# -----------------------------------------------------

@when("I evaluate 300 transactions")
def step_eval_300(context):
    txs = context.env.batch(300)
    states = np.stack([tx.state for tx in txs])
    labels = np.array([int(tx.is_fraud) for tx in txs])
    preds, _ = context.agent.predict_batch(states)

    tp = ((preds == 1) & (labels == 1)).sum()
    fn = ((preds == 0) & (labels == 1)).sum()

    context.recall = tp / (tp + fn + 1e-8)


@then("recall should be greater than 0.6")
def step_recall_check(context):
    assert context.recall > 0.6


# -----------------------------------------------------
# PPO Stability
# -----------------------------------------------------

@then("the KL divergence should be less than 0.2")
def step_kl_check(context):
    latest = context.tracker.latest
    assert latest["mean_kl"] < 0.2


