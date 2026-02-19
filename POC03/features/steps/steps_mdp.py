"""
POC-03: BDD Step Definitions for MDP Modeling

This file connects Gherkin feature scenarios
to the TaxiMDP implementation.

We validate:

1. State space size
2. Action space size
3. Deterministic transitions
4. Transition consistency
5. Reward correctness

These tests ensure the Taxi-v3 environment
correctly satisfies the formal MDP definition.
"""

from behave import given, when, then
from mdp_model import TaxiMDP


# ============================================================
# Shared Setup
# ============================================================

@given("the Taxi MDP model is initialized")
def step_initialize_mdp(context):
    """
    Instantiate the TaxiMDP model.

    This extracts:
        • State space (S)
        • Action space (A)
        • Transition model (P)
        • Reward function (R)
        • Discount factor (γ)
    """
    context.mdp = TaxiMDP(gamma=0.9)


# ============================================================
# Scenario 1 — State Space Validation
# ============================================================

@then("the number of states should be 500")
def step_validate_state_space(context):
    """
    Taxi-v3 has:

        25 taxi positions
        5 passenger states
        4 destinations

    Total = 25 × 5 × 4 = 500
    """
    assert context.mdp.get_state_space_size() == 500


# ============================================================
# Scenario 2 — Action Space Validation
# ============================================================

@then("the number of actions should be 6")
def step_validate_action_space(context):
    """
    Taxi-v3 defines 6 discrete actions:

        0: South
        1: North
        2: East
        3: West
        4: Pickup
        5: Dropoff
    """
    assert context.mdp.get_action_space_size() == 6


# ============================================================
# Scenario 3 — Determinism Check
# ============================================================

@then("the environment should be deterministic")
def step_validate_determinism(context):
    """
    Taxi-v3 is deterministic:

    For every (state, action) pair,
    there is exactly one next state
    with probability = 1.
    """
    assert context.mdp.is_deterministic() is True


# ============================================================
# Scenario 4 — Transition Consistency
# ============================================================

@when("the same state-action pair is queried multiple times")
def step_query_transition(context):
    """
    Query the same (state, action) multiple times
    to confirm deterministic behavior.
    """
    state = 100
    action = 2  # East

    results = []

    for _ in range(5):
        next_state, reward, done = context.mdp.sample_transition(state, action)
        results.append(next_state)

    context.transition_results = results


@then("the resulting next state should always be identical")
def step_validate_consistency(context):
    """
    All next states should be identical
    in a deterministic environment.
    """
    first_state = context.transition_results[0]

    for state in context.transition_results:
        assert state == first_state


# ============================================================
# Scenario 5 — Reward Validation
# ============================================================

@when("a legal dropoff transition is inspected")
def step_inspect_dropoff(context):
    """
    Find a state where dropoff action gives +20 reward.

    We iterate through transitions to locate
    a successful dropoff.
    """
    mdp = context.mdp

    found = False

    for state in range(mdp.get_state_space_size()):
        transitions = mdp.get_transition(state, 5)  # Dropoff action

        prob, next_state, reward, done = transitions[0]

        if reward == 20:
            context.dropoff_reward = reward
            found = True
            break

    assert found is True, "No successful dropoff found."


@then("the reward should be 20")
def step_validate_dropoff_reward(context):
    """
    Successful dropoff must yield +20 reward
    according to Taxi-v3 specification.
    """
    assert context.dropoff_reward == 20
