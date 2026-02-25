Feature: Markov Decision Process Modeling for Taxi-v3

  The Taxi-v3 environment should satisfy the formal
  definition of a Markov Decision Process (MDP).

  Scenario: State space size validation
    Given the Taxi MDP model is initialized
    Then the number of states should be 500

  Scenario: Action space size validation
    Given the Taxi MDP model is initialized
    Then the number of actions should be 6

  Scenario: Deterministic transition validation
    Given the Taxi MDP model is initialized
    Then the environment should be deterministic

  Scenario: Transition consistency check
    Given the Taxi MDP model is initialized
    When the same state-action pair is queried multiple times
    Then the resulting next state should always be identical

  Scenario: Reward structure validation
    Given the Taxi MDP model is initialized
    When a legal dropoff transition is inspected
    Then the reward should be 20
