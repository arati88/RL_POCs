import os
import json
from behave import given, when, then

# Go up 4 levels to reach RL root
BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)

RESULT_FILE = os.path.join(BASE_DIR, "poc08_results.json")


@given('the POC-08 results JSON file exists')
def step_check_file(context):
    print("Looking at:", RESULT_FILE)
    assert os.path.exists(RESULT_FILE), f"Results JSON file not found at {RESULT_FILE}"

@when('I load the evaluation metrics')
def step_load_metrics(context):
    with open(RESULT_FILE, "r") as f:
        context.data = json.load(f)

@then('precision should be greater than {threshold:f}')
def step_check_precision(context, threshold):
    precision = context.data["precision"]
    assert precision > threshold, f"Precision {precision} is below {threshold}"

@then('recall should be greater than {threshold:f}')
def step_check_recall(context, threshold):
    recall = context.data["recall"]
    assert recall > threshold, f"Recall {recall} is below {threshold}"

@then('F1 score should be greater than {threshold:f}')
def step_check_f1(context, threshold):
    f1 = context.data["f1_score"]
    assert f1 > threshold, f"F1 {f1} is below {threshold}"

@then('accuracy should be greater than {threshold:f}')
def step_check_accuracy(context, threshold):
    accuracy = context.data["accuracy"]
    assert accuracy > threshold, f"Accuracy {accuracy} is below {threshold}"

@when('I check convergence analysis')
def step_check_convergence(context):
    with open(RESULT_FILE, "r") as f:
        context.data = json.load(f)

@then('convergence status should be "{status}"')
def step_validate_status(context, status):
    actual_status = context.data["convergence_status"]
    assert actual_status == status, f"Expected {status}, got {actual_status}"

@then('final average reward should be positive')
def step_reward_positive(context):
    reward = context.data["final_avg_reward"]
    assert reward > 0, "Final reward is not positive"

@when('I analyze reward history')
def step_analyze_rewards(context):
    with open(RESULT_FILE, "r") as f:
        context.data = json.load(f)

@then('the final average reward should be greater than the initial average reward')
def step_compare_rewards(context):
    initial = context.data["final_avg_reward"] - context.data["reward_std_last_20"]
    final = context.data["final_avg_reward"]
    assert final > initial, "Reward did not improve over training"